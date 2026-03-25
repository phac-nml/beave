"""
Re-implementation of mcluster
"""

import sys
import logging
import math
import pathlib as p
import typing as t
from enum import StrEnum, Enum


import dist_mat as dm
import scipy as sp
import polars as pl
import numpy as np
from numpy import typing as npt

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class LinkageMetrics(StrEnum):
    SINGLE = "single"
    AVERAGE = "average"
    COMPLETE = "complete"


class BranchLengths(StrEnum):
    PATRISTIC = "patristic"
    COPHENETIC = "cophenetic"


class LinkageMatrixFields(Enum):
    """
    From the scipy docs:

    A by 4 matrix Z is returned. At the i-th iteration, clusters with indices
    Z[i, 0] and Z[i, 1] are combined to form cluster n+1. A cluster with an
    index less than n corresponds to one of the original observations. The
    distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2].
    The fourth value Z[i, 3] represents the number of original observations in
    the newly formed cluster.

    """

    OBS1 = 0  # Observation 1, can be a sample or formed cluster
    OBS2 = (
        1  # Observation 2, this is a sample or cluster that is being combined wit OBS 1
    )
    DISTANCE = 2  # This is the distance between OBS1 and OBS2
    NUM_OBSERVATIONS = (
        3  # This value represents the number of orignal observations in the new cluster
    )


MISSING_VALUE = np.uint32(0)
REPLACE_CHARS = {
    "?": MISSING_VALUE,
    " ": MISSING_VALUE,
    "-": MISSING_VALUE,
    "": MISSING_VALUE,
    "_": MISSING_VALUE,
    "0": MISSING_VALUE,
}  # mappings to replace fields with zeroes


def _scipy_tree_to_newick_list(
    node: sp.cluster.hierarchy.ClusterNode | None,
    newick: list[str],
    parentdist: float,
    leaf_names: list[str],
) -> list[str]:
    """Construct Newick tree from SciPy hierarchical clustering ClusterNode

    This is a recursive function to help build a Newick output string from a scipy.cluster.hierarchy.to_tree input with
    user specified leaf node names.

    Notes:
        This function is meant to be used with `to_newick`

    Args:
        node (scipy.cluster.hierarchy.ClusterNode): Root node is output of scipy.cluster.hierarchy.to_tree from hierarchical clustering linkage matrix
        parentdist (float): Distance of parent node of `node`
        newick (list of string): Newick string output accumulator list which needs to be reversed and concatenated (i.e. `''.join(newick)`) for final output
        leaf_names (list of string): Leaf node names

    Returns:
        (list of string): Returns `newick` list of Newick output strings
    """

    if node is None:
        return newick

    if node.is_leaf():
        return newick + [f"{leaf_names[node.id]}:{parentdist - node.dist}"]

    if len(newick) > 0:
        newick.append(f"):{parentdist - node.dist}")
    else:
        newick.append(");")
    newick = _scipy_tree_to_newick_list(node.get_left(), newick, node.dist, leaf_names)
    newick.append(",")
    newick = _scipy_tree_to_newick_list(node.get_right(), newick, node.dist, leaf_names)
    newick.append("(")
    return newick


def to_newick(tree: sp.cluster.hierarchy.ClusterNode, leaf_names: list[str]) -> str:
    """Newick tree output string from SciPy hierarchical clustering tree

    Convert a SciPy ClusterNode tree to a Newick format string.
    Use scipy.cluster.hierarchy.to_tree on a hierarchical clustering linkage matrix to create the root ClusterNode for the `tree` input of this function.

    Args:
        tree (scipy.cluster.hierarchy.ClusterNode): Output of scipy.cluster.hierarchy.to_tree from hierarchical clustering linkage matrix
        leaf_names (list of string): Leaf node names

    Returns:
        (string): Newick output string
    """
    newick_list = _scipy_tree_to_newick_list(tree, [], tree.dist, leaf_names)
    return "".join(newick_list[::-1])


def subset_columns(profiles: pl.DataFrame, columns_keep: p.Path) -> pl.DataFrame:
    """
    Subset only the required columns to use for distance matrix calculation
    """
    columns_to_keep: set[str] | None = None
    with columns_keep.open("r") as ck:
        columns_to_keep = set([line.strip() for line in ck.readlines()])

    sample_col = profiles.columns[0]
    profile_cols = set(profiles.columns[1:])  # keep all columns but first
    columns_keep_set = list(profile_cols & columns_to_keep)
    return profiles.select(sample_col, *columns_keep_set)


def read_input_profiles(
    input: p.Path, columns_keep: p.Path | None, delimiter: str, threads: int
) -> pl.DataFrame:
    """
    Read the allelic profiles and perform any filtering and conversions required.

    Check for nulls in sample id column and enforces uniqueness


    Polars mangles columns with ptotential duplicate names on loading, potential bug*
    """

    profiles = pl.read_csv(
        input,
        separator=delimiter,
        n_threads=threads,
        has_header=True,
        raise_if_empty=True,
        missing_utf8_is_empty_string=True,
        infer_schema=False,
    )
    if profiles.shape[1] <= 1:
        logger.critical(
            f"Only {profiles.shape[1]} in allele profiles, you need atleast two loci columns."
        )
        raise pl.exceptions.ShapeError(
            f"Only {profiles.shape[1]} in allele profiles, you need atleast two loci columns."
        )

    if profiles.shape[0] <= 1:
        logger.critical(
            f"Only {profiles.shape[0]} in allele profiles, you need atleast two sample rows."
        )
        raise pl.exceptions.RowsError(
            f"Only {profiles.shape[0]} in allele profiles, you need atleast two sample rows."
        )

    # Cannot use null_count in polars for this, as we convert all null values into empty strings
    if profiles.select((pl.nth(0) == "").sum())[0, 0] >= 1:
        logger.critical(
            "Missing values identified in left most column (ID column), left most column can have no missing values."
        )
        raise pl.exceptions.RowsError(
            "Missing values identified in left most column (ID column), left most column can have no missing values."
        )

    if not profiles.select(pl.nth(0)).is_unique().all():
        logger.critical(
            "Duplicate values identified in left most column (ID column), leftmost column can have no missing values."
        )
        raise pl.exceptions.DuplicateError(
            "Duplicate values identified in left most column (ID column), leftmost column can have no missing values."
        )

    if columns_keep is not None:
        profiles = subset_columns(profiles, columns_keep)

    return profiles


def filter_rows(profiles: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """
    Remove rows from the dataframe that have are missing more than
    the thresholds set limit for missing data.


    """
    if threshold == 0.00:
        return profiles

    number_of_columns = profiles.width - 1  # -1 to ignore the labels column
    threshold_columns = math.ceil(number_of_columns * threshold)
    logger.info(
        f"Setting filter threshold to excluded columns missing {threshold_columns} or more loci."
    )
    rows_before_filtering = profiles.height
    profiles = profiles.filter(
        pl.sum_horizontal(
            pl.all().exclude(profiles.columns[0]) == MISSING_VALUE
        )  # select all columns but first id col
        < threshold_columns
    )

    logger.info(
        f"Removed {rows_before_filtering - profiles.height} rows after filtering."
    )

    return profiles


def transform_data(profiles: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """
    Transform the dataframe of profiles by hashing the entries, converting missing allele
    charactars to zeroes and filtering rows.
    """

    # Can add additonal qc filtering here
    data_columns = profiles.columns[1:]  # only apply functions to loci columns
    profiles = profiles.with_columns(
        pl.all()
        .exclude(profiles.columns[0])  # skip id column
        .replace(REPLACE_CHARS)
    )

    profiles = profiles.with_columns(
        [
            pl.when(pl.col(i) != str(MISSING_VALUE))
            .then(pl.col(i).hash(42, 42, 42, 42))
            .otherwise(pl.lit(0))
            for i in data_columns
        ]
    )

    profiles = filter_rows(profiles, threshold)
    return profiles


def prep_data(profiles: pl.DataFrame, threshold: float) -> npt.NDArray:
    """
    Prepare profiles for computation by the the calc_dists function of dist_mat.
    """

    data_columns = profiles.columns[1:]  # only apply functions to loci columns
    profiles = transform_data(profiles, threshold)

    profiles_numpy = profiles.select([pl.col(i) for i in data_columns]).to_numpy()
    profiles_numpy = profiles_numpy.astype(
        np.uint32
    )  # convert hashes to 32 bit representation for dist_mat
    return profiles_numpy


def compute_dists(
    profiles: pl.DataFrame,
    count_missing: bool,
    scaled: bool,
    threads: int,
    filter_threshold: float = 0.0,
) -> npt.NDArray:
    """
    Compute the 1D array required by scipy for generation of the linkage matrix.
    """
    prepared_profiles = prep_data(profiles, filter_threshold)
    distances = dm.calc_dists(prepared_profiles, threads, scaled, count_missing)
    return distances


def comp_linkage_matrix(
    profiles_computed: npt.NDArray, linkage_method: str
) -> npt.NDArray:
    linkage = sp.cluster.hierarchy.linkage(profiles_computed, method=linkage_method)  # type: ignore[arg-type]
    return linkage


def assign_clusters(
    linkage: npt.NDArray, thresholds: list[int | float], labels: list[str]
) -> pl.DataFrame:
    """
    Generate the new cluster membership code for all samples at each threshold.

    Thresholds are assumed to be sorted in descending order e.g [10.0, 5.0, 1.0]
    """

    data_to_populate = [pl.Series(name="SampleID", values=labels)]
    cols_concat = []

    for threshold in thresholds:
        col_name = f"level_{str(threshold)}"
        data_to_populate.append(
            pl.Series(
                name=col_name,
                values=sp.cluster.hierarchy.fcluster(
                    linkage, threshold, criterion="distance"
                ),
                dtype=pl.UInt32,
            )
        )

        cols_concat.append(col_name)

    outputs = pl.DataFrame(data_to_populate, orient="col")
    outputs = outputs.with_columns(
        pl.concat_str(cols_concat, separator=".").alias("denovo_address")
    )

    return outputs


def convert_branch_lengths(
    linkage_matrix: npt.NDArray, bl_type: BranchLengths
) -> npt.NDArray:
    """
    Convert linkage matrix to to cophenetic distance if needed.
    """
    if bl_type == BranchLengths.COPHENETIC:
        for row in linkage_matrix:
            row[LinkageMatrixFields.DISTANCE.value] *= 2
    return linkage_matrix


def mcluster(
    input: p.Path,
    delimiter: str,
    thresholds: list[float],
    method: str,
    n_threads: int,
    columns: p.Path | None,
    count_missing: bool,
    scaled: bool,
    tree_output: p.Path,
    cluster_outputs: p.Path,
    tree_distances: BranchLengths,
    filter_threshold: float,
    *args: list[t.Any],
    **kwargs: dict[t.Any, t.Any],
) -> None:
    """
    Main runner function for mcluster.
    """

    profiles = read_input_profiles(input, columns, delimiter, n_threads)
    logger.info("Loaded profiles")
    distances = compute_dists(
        profiles, count_missing, scaled, n_threads, filter_threshold
    )
    logger.info("Computed distances")
    linkages = comp_linkage_matrix(distances, method)
    logger.info("Computed linkage matrix")
    thresholds.sort(reverse=True)
    logger.info(f"Thresholds being used for generating linkages: {thresholds}")

    sample_names = profiles.select(pl.nth(0)).to_series().to_list()
    # write out the tree
    cluster_memberships = assign_clusters(linkages, thresholds, sample_names)
    logger.info("assigned clusters")

    sys.setrecursionlimit(4000)  # raise recursion limit for generating the tree
    linkages = convert_branch_lengths(
        linkages, tree_distances
    )  # convert branch lengths for tree display if needed
    tree = sp.cluster.hierarchy.to_tree(linkages)
    newick = to_newick(tree, sample_names)

    with tree_output.open("w") as to:
        to.write(newick)
    logger.info(f"Wrote newick tree to: {str(tree_output)}")

    cluster_memberships.write_csv(
        cluster_outputs, separator=delimiter, include_header=True
    )
    logger.info(f"Wrote cluster memberships to: {str(cluster_outputs)}")
