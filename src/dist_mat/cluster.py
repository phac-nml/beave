"""Re-implementation of mcluster."""

import logging
import math
import sys
import typing as t
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path

import numpy as np
import polars as pl
import scipy
from numpy import typing as npt

import dist_mat as dm

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ValueLeavesError(Exception):
    """Exception for raising a value error for un-equal numbers of objects."""

    def __init__(self, n_leaves: int, n_objects: int) -> None:
        """ValueError for unequal numbers of leaves and sample names."""
        super().__init__(f"Expected {n_objects} leaf names, got {n_leaves}.")


class AllColumnsFilteredError(Exception):
    """Exception raised if all rows are filtered."""

    def __init__(self, threshold: float) -> None:
        """Error raised if all column values removed."""
        super().__init__(f"All data removed after filtering at: {threshold}.")


class LinkageMetric(StrEnum):
    """Linkage options that can be passed to scipy."""

    SINGLE = "single"
    AVERAGE = "average"
    COMPLETE = "complete"


class BranchLengthType(StrEnum):
    """Branch length types used for converting tree branch metrics."""

    PATRISTIC = "patristic"
    COPHENETIC = "cophenetic"


@dataclass(slots=True)
class ClusterArguments:
    """CLI arguments for clustering program.

    Disabling the pylint warning for too-many-instance-attributes
    as the number of attributes seems appropriate here for the purpose
    the class serves.
    """

    # pylint: disable=too-many-instance-attributes

    input_file: Path
    delimiter: str
    thresholds: list[float]
    method: str
    cores: int
    columns_path: Path | None
    count_missing: bool
    scaled: bool
    tree_output: Path
    cluster_outputs: Path
    branch_length_type: BranchLengthType
    filter_threshold: float


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


MISSING_VALUE = np.uint32(0)
REPLACE_CHARS = {
    "?": MISSING_VALUE,
    " ": MISSING_VALUE,
    "-": MISSING_VALUE,
    "": MISSING_VALUE,
    "_": MISSING_VALUE,
    "0": MISSING_VALUE,
}  # mappings to replace fields with zeroes


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


def subset_columns(profiles: pl.DataFrame, columns_path: Path) -> pl.DataFrame:
    """Subset only the required columns to use for distance matrix calculation."""
    columns: set[str] | None = None
    with columns_path.open("r") as columns_file:
        columns = {line.strip() for line in columns_file.readlines()}

    sample_col = profiles.columns[0]
    profile_cols = set(profiles.columns[1:])  # keep all columns but first
    columns_keep = list(profile_cols & columns)
    return profiles.select(sample_col, *columns_keep)


def read_input_profiles(
    input_file: Path, columns_keep_path: Path | None, delimiter: str, threads: int
) -> pl.DataFrame:
    """Read the allelic profiles and perform any filtering and conversions required.

    Check for nulls in sample id column and enforces uniqueness


    Polars mangles columns with potential duplicate names on loading, potential bug*
    """
    profiles = pl.read_csv(
        input_file,
        separator=delimiter,
        n_threads=threads,
        has_header=True,
        raise_if_empty=True,
        missing_utf8_is_empty_string=True,
        infer_schema=False,
    )
    if profiles.shape[1] <= 1:
        err_string = (
            f"Only {profiles.shape[1]} in allele profiles, you need atleast two loci columns."
        )

        logger.critical(err_string)
        raise pl.exceptions.ShapeError(err_string)

    if profiles.shape[0] <= 1:
        err_string = (
            f"Only {profiles.shape[0]} in allele profiles were loaded, but atleast two "
            f"profiles must be provided."
        )
        logger.critical(err_string)
        raise pl.exceptions.RowsError(err_string)

    # Cannot use null_count in polars for this, as we convert all null values into empty strings
    if profiles.select((pl.nth(0) == "").sum())[0, 0] >= 1:
        err_string = (
            "Missing values identified in left most column (ID column), left most column "
            "can have no missing values."
        )
        logger.critical(err_string)
        raise pl.exceptions.RowsError(err_string)

    if not profiles.select(pl.nth(0)).is_unique().all():
        err_string = (
            "Duplicate values identified in left most column (ID column), leftmost column"
            " can have no missing values."
        )
        logger.critical(err_string)
        raise pl.exceptions.DuplicateError(err_string)

    if columns_keep_path:
        profiles = subset_columns(profiles, columns_keep_path)

    return profiles


def filter_rows(profiles: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Remove rows missing a certain percentage of data.

    Remove rows from the dataframe that have are missing more than the thresholds set limit for
    missing data.
    """
    hundred_percent: float = 1.0
    if threshold == hundred_percent:
        return profiles

    number_of_columns = profiles.width - 1  # -1 to ignore the labels column
    threshold_columns = math.floor(number_of_columns * threshold)
    logger.info(
        "Setting filter threshold to excluded columns missing % or more loci.",
        threshold_columns,
    )
    rows_before_filtering = profiles.height
    profiles = profiles.filter(
        pl.sum_horizontal(
            pl.all().exclude(profiles.columns[0]) == MISSING_VALUE
        )  # select all columns but first id col
        <= threshold_columns
    )
    if profiles.is_empty():
        raise AllColumnsFilteredError(threshold)

    logger.info("Removed %s rows after filtering.", rows_before_filtering - profiles.height)

    return profiles


def transform_data_hashes(profiles: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Return data prepared for calc_dists.

    Transform the dataframe of profiles by hashing the entries, converting missing allele
    charactars to zeroes and filtering rows.
    """
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


def transform_data(profiles: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Return data prepared for calc_dists.

    Transform the dataframe of profiles by creating a look up table to cast values to integers,
    converting missing allele charactars to zeroes and filtering rows.
    """
    # Create mapping instead of using hashes
    values_columns = 1
    unique_values = (
        profiles.with_columns(pl.all().exclude(profiles.columns[0]))
        .unpivot()
        .to_series(values_columns)
        .unique()
        .to_list()
    )

    char_mapping = (
        {  # start mapping at 1, as 0 is used for missing values and add one to not miss values
            value: idx
            for value, idx in zip(
                unique_values, np.arange(1, len(unique_values) + 1, dtype=np.uint32)
            )
        }
        | REPLACE_CHARS
    )  # Create new dictionary, REPLACE_CHARS keys overwrite those in new dictionary

    profiles = profiles.with_columns(
        pl.all()
        .exclude(profiles.columns[0])  # skip id column
        .replace(char_mapping)
        .cast(pl.UInt32)  # strict cast will throw an error if any overflow occurs
    )

    profiles = filter_rows(profiles, threshold)
    return profiles


def prep_data(
    profiles: pl.DataFrame,
    threshold: float,
    transformation_func: t.Callable[[pl.DataFrame, float], pl.DataFrame],
) -> npt.NDArray:
    """Prepare profiles for computation by the the calc_dists function of dist_mat."""
    data_columns = profiles.columns[1:]  # only apply functions to loci columns
    profiles = transformation_func(profiles, threshold)
    profiles_numpy = profiles.select([pl.col(i) for i in data_columns]).to_numpy().astype(np.uint32)
    return profiles_numpy


def compute_dists(
    profiles: pl.DataFrame,
    count_missing: bool,
    scaled: bool,
    threads: int,
    filter_threshold: float = 1.0,
) -> npt.NDArray:
    """Compute the 1D array required by scipy for generation of the linkage matrix."""
    prepared_profiles = prep_data(profiles, filter_threshold, transform_data)
    distances = dm.calc_dists(prepared_profiles, threads, scaled, count_missing)
    return distances


def compute_linkage_matrix(profiles_computed: npt.NDArray, linkage_method: str) -> npt.NDArray:
    """Use scipy to compute a linkage matrix from the calculated distances.

    profiles_computed refers to a 1D condensed array generated by calc_dists
    """
    linkage = scipy.cluster.hierarchy.linkage(profiles_computed, method=linkage_method)  # type: ignore[arg-type]
    return linkage


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

    outputs = pl.DataFrame(data_to_populate, orient="col")
    outputs = outputs.with_columns(
        pl.concat_str(cols_concat, separator=".").alias("denovo_address")
    )

    return outputs


def convert_branch_lengths(
    linkage_matrix: npt.NDArray, branch_length_type: BranchLengthType
) -> npt.NDArray:
    """Convert linkage matrix to to cophenetic distance if needed."""
    if branch_length_type == BranchLengthType.PATRISTIC:
        for row in linkage_matrix:
            row[LinkageMatrixFields.DISTANCE.value] *= 0.5
    return linkage_matrix


def cluster(cluster_args: ClusterArguments) -> None:
    """Runner function of cluster."""
    profiles = read_input_profiles(
        cluster_args.input_file,
        cluster_args.columns_path,
        cluster_args.delimiter,
        cluster_args.cores,
    )
    logger.info("Loaded profiles")
    distances = compute_dists(
        profiles,
        cluster_args.count_missing,
        cluster_args.scaled,
        cluster_args.cores,
        cluster_args.filter_threshold,
    )
    logger.info("Computed distances")
    linkages = compute_linkage_matrix(distances, cluster_args.method)
    logger.info("Computed linkage matrix")
    cluster_args.thresholds.sort(reverse=True)
    logger.info("Thresholds being used for generating linkages: %s", cluster_args.thresholds)

    sample_names = profiles.select(pl.nth(0)).to_series().to_list()
    # write out the tree
    cluster_memberships = assign_clusters(linkages, cluster_args.thresholds, sample_names)
    logger.info("assigned clusters")

    linkages = convert_branch_lengths(
        linkages, cluster_args.branch_length_type
    )  # convert branch lengths for tree display if needed
    newick = linkage_matrix_to_nwk(linkages, sample_names)

    with cluster_args.tree_output.open("w") as to:
        to.write(newick)
    logger.info("Wrote newick tree to: %s", str(cluster_args.tree_output))

    cluster_memberships.write_csv(
        cluster_args.cluster_outputs,
        separator=cluster_args.delimiter,
        include_header=True,
    )
    logger.info("Wrote cluster memberships to: %s", str(cluster_args.cluster_outputs))
