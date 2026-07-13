"""Re-implementation of mcluster."""

import asyncio
from enum import Enum
from pathlib import Path

import numpy as np
import polars as pl
import scipy
from numpy import typing as npt

import beave
from beave.declarations import BranchType, ClusterArguments
from beave.log import init_logger
from beave.transform_data import (
    prep_data,
    read_input_profiles,
    subset_columns,
    transform_data_categorical_encoding,
)

logger = init_logger(__name__)


class ValueLeavesError(Exception):
    """Exception for raising a value error for un-equal numbers of objects."""

    def __init__(self, n_leaves: int, n_objects: int) -> None:
        """ValueError for unequal numbers of leaves and sample names."""
        super().__init__(f"Sorry, expected {n_objects} leaf names, got {n_leaves}.")


class LinkageMatrixFields(Enum):
    """From the scipy docs.

    A by 4 matrix Z is returned. At the i-th iteration, clusters with indices
    Z[i, 0] and Z[i, 1] are combined to form cluster n+1. A cluster with an
    index less than n corresponds to one of the original observations. The
    distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2].
    The fourth value Z[i, 3] represents the number of original observations in
    the newly formed cluster.

    """

    OBS1 = 0  # Observation 1, can be a sample or formed cluster
    OBS2 = 1  # Observation 2, this is a sample or cluster that is being combined wit OBS 1
    DISTANCE = 2  # This is the distance between OBS1 and OBS2
    NUM_OBSERVATIONS = (
        3  # This value represents the number of orignal observations in the new cluster
    )


def linkage_matrix_to_nwk(linkage_matrix: npt.NDArray, sample_ids: list[str]) -> str:
    """Code is taken from an old Scipy PR.

    I did not come up with this code below is has been adapated from a scipy PR here:
    https://github.com/scipy/scipy/pull/17329/changes

    The inputs are the linkage matrix from scipy, and the sample_ids correspond to
    the alleleic profiles.
    """
    n_objects: int = linkage_matrix.shape[0] + 1
    n_leaves: int = len(sample_ids)
    if n_objects != n_leaves:
        raise ValueLeavesError(n_leaves, n_objects)

    newick_intermediates: list[str | None] = sample_ids + [None] * linkage_matrix.shape[0]
    cluster_dists: list[np.float64] = [np.float64(0)] * (n_objects + linkage_matrix.shape[0])
    for i, row in enumerate(linkage_matrix):
        dist: np.float64 = row[LinkageMatrixFields.DISTANCE.value]
        fi: int = int(row[LinkageMatrixFields.OBS1.value])
        fj: int = int(row[LinkageMatrixFields.OBS2.value])
        cdi: np.float64 = dist - cluster_dists[fi]
        cdj: np.float64 = dist - cluster_dists[fj]
        newick_subtree: str = f"({newick_intermediates[fi]}:{cdi},{newick_intermediates[fj]}:{cdj})"
        newick_intermediates[i + n_objects] = newick_subtree
        cluster_dists[i + n_objects] = dist
        newick_intermediates[fi] = None
        newick_intermediates[fj] = None

    return newick_intermediates[linkage_matrix.shape[0] - 1 + n_objects] + ";"


def prepare_matrix(array: npt.NDArray, columns: pl.Series) -> pl.DataFrame:
    """Prepare the matrix object as a polars array."""
    square_matrix: npt.NDArray = scipy.spatial.distance.squareform(array)
    output_df: pl.DataFrame = pl.from_numpy(square_matrix, schema=columns.to_list())
    output_df = output_df.insert_column(0, columns)
    return output_df


def compute_dists(
    profiles: pl.DataFrame,
    cluster_args: ClusterArguments,
) -> tuple[npt.NDArray, pl.Series]:
    """Compute the 1D array required by scipy for generation of the linkage matrix."""
    profiles_array, filtered_profiles = prep_data(
        profiles, cluster_args.filter_threshold, transform_data_categorical_encoding, cluster_args
    )
    logger.debug("Tranformed data for computation in C++ sub-routine.")
    distances = beave.calc_dists(
        profiles_array,
        cluster_args.cores,
        cluster_args.normalize_distance,
        cluster_args.count_missing,
    )
    logger.debug("Finished C++ sub-routine.")
    return distances, filtered_profiles


def dists_to_matrix(square_array: pl.DataFrame, seperator: str, output_file: Path) -> None:
    """Write square array to file as matrix in a seperate co-routine.

    This function will run sequentially with the rest of the program
    e.g. not truly run in parallel. As GIL'less python becomes the standard
    with future python releases this function will be able to take advantage
    of the ability to run in parallel in the near future.

    This process and computation of the linkage files currently do run in seperate threads,
    however how context switching happens is not quite clear yet. Running the two functions
    in seperate threads has not added significant overhead to the program in benchmarking
    however.
    """
    logger.debug(f"Beginning write of matrix to {output_file}.")
    square_array.write_csv(output_file, separator=seperator, include_header=True)
    logger.info("Finished writing distance matrix to file.")


def compute_linkage_matrix(profiles_computed: npt.NDArray, linkage: str) -> npt.NDArray:
    """Use scipy to compute a linkage matrix from the calculated distances.

    profiles_computed refers to a 1D condensed array generated by calc_dists
    """
    logger.info(f"Calculating {linkage} linkage.")
    linkage_out = scipy.cluster.hierarchy.linkage(profiles_computed, method=linkage)  # type: ignore[arg-type]
    logger.info("Finished computing linkage matrix.")
    return linkage_out


def assign_clusters(
    linkage: npt.NDArray, thresholds: list[int | float], labels: list[str]
) -> pl.DataFrame:
    """Generate the new cluster membership code for all samples at each threshold.

    Thresholds are assumed to be sorted in descending order e.g [10.0, 5.0, 1.0]
    """
    data_to_populate = [pl.Series(name="SampleID", values=labels)]
    cols_concat = []

    for threshold in thresholds:
        col_name = f"level_{threshold!s}"
        data_to_populate.append(
            pl.Series(
                name=col_name,
                values=scipy.cluster.hierarchy.fcluster(linkage, threshold, criterion="distance"),
                dtype=pl.UInt32,
            )
        )
        cols_concat.append(col_name)
        logger.debug(f"Calculated flat clusters for threshold: {threshold}")

    outputs = pl.DataFrame(data_to_populate, orient="col")
    outputs = outputs.with_columns(
        pl.concat_str(cols_concat, separator=".").alias("denovo_address")
    )

    return outputs


def convert_branch_lengths(linkage_matrix: npt.NDArray, branch_type: BranchType) -> npt.NDArray:
    """Convert linkage matrix to to cophenetic distance if needed."""
    if branch_type == BranchType.PATRISTIC:
        for row in linkage_matrix:
            row[LinkageMatrixFields.DISTANCE.value] *= 0.5
        logger.info("Converted branch lengths to patristic distances.")
    return linkage_matrix


def create_matrix(
    cluster_args: ClusterArguments,
    distances: npt.NDArray,
    profile_names: pl.Series,
    output_extension: str,
    tg: asyncio.TaskGroup | None,
) -> asyncio.Task[None] | None:
    """Write a pairwise distance matrix to the output directory."""
    logger.info("Preparing distance matrix for write to file.")
    matrix_output: Path = cluster_args.output_directory / f"matrix.{output_extension}"
    logger.debug("Fromatting matrix.")
    square_matrix: pl.DataFrame = prepare_matrix(distances, profile_names)
    logger.debug("Finished preparing distance matrix for output.")

    if tg is None:
        dists_to_matrix(square_matrix, cluster_args.delimiter, matrix_output)
        return None

    return tg.create_task(  # pyright: ignore[reportAssignmentType]
        asyncio.to_thread(dists_to_matrix, square_matrix, cluster_args.delimiter, matrix_output)
    )


async def cluster(cluster_args: ClusterArguments, output_extension: str) -> None:
    """Runner function of cluster."""
    profiles: pl.DataFrame = read_input_profiles(
        cluster_args.input_file,
        cluster_args.delimiter,
        cluster_args.cores,
    )
    logger.info("Loaded profiles")

    if cluster_args.columns_path:
        logger.info("Subsetting columns.")
        profiles = subset_columns(profiles, cluster_args.columns_path, None)
        logger.debug("Finished subsetting columns.")

    distances, profile_names = compute_dists(
        profiles,
        cluster_args,
    )

    logger.info("Computed distances.")
    if cluster_args.matrix_only:
        create_matrix(cluster_args, distances, profile_names, output_extension, tg)

    async with asyncio.TaskGroup() as tg:
        linkages_task = tg.create_task(
            asyncio.to_thread(compute_linkage_matrix, distances, cluster_args.linkage_method)
        )
        matrix_task: asyncio.Task[None] | None = None
        if cluster_args.matrix:
            matrix_task = create_matrix(
                cluster_args, distances, profile_names, output_extension, tg
            )

    linkages = linkages_task.result()
    if matrix_task:
        await matrix_task

    cluster_args.thresholds.sort(reverse=True)
    logger.info("Thresholds being used for generating linkages: %s", cluster_args.thresholds)

    sample_names: list[str] = profile_names.to_list()
    # write out the tree
    cluster_memberships = assign_clusters(linkages, cluster_args.thresholds, sample_names)
    logger.info("Assigned clusters.")

    linkages = convert_branch_lengths(
        linkages, cluster_args.branch_type
    )  # convert branch lengths for tree display if needed
    newick = linkage_matrix_to_nwk(linkages, sample_names)

    tree_output: Path = cluster_args.output_directory / "tree.nwk"
    with tree_output.open("w") as to:
        to.write(newick)
    logger.info("Wrote newick tree to: %s", str(tree_output))

    cluster_output: Path = cluster_args.output_directory / f"clusters.{output_extension}"
    cluster_memberships.write_csv(
        cluster_output,
        separator=cluster_args.delimiter,
        include_header=True,
    )
    logger.info("Wrote cluster memberships to: %s", str(cluster_output))
