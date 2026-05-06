"""Internal functions used for data loading and transformation."""

import math
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from beave.log import init_logger

logger = init_logger(__name__)


class DistanceTypes(StrEnum):
    """String storage for the distance types used."""

    HAMMING = "hamming"
    NORMALIZED = "normalized"


class AllColumnsFilteredError(Exception):
    """Exception raised if all rows are filtered."""

    def __init__(self, threshold: float) -> None:
        """Error raised if all column values removed."""
        super().__init__(f"Sorry, all data removed after filtering at: {threshold}.")


class MissingIDValueError(Exception):
    """Exception raised if missing an ID field in input data."""

    def __init__(self, error: str) -> None:
        """Error raised if missing a sample ID value."""
        super().__init__(error)


MISSING_VALUE = np.uint32(0)
REPLACE_CHARS = {
    "?": None,
    " ": None,
    "-": None,
    "": None,
    "_": None,
    "0": None,
}  # mappings to replace fields with zeroes


def subset_columns(
    profiles: pl.DataFrame, /, columns_path: Path | None, columns_keep: set[str] | None
) -> pl.DataFrame:
    """Subset only the required columns to use for distance matrix calculation."""
    columns: set[str]

    if columns_path is None and columns_keep is None:
        error_msg = (
            "Sorry, subset columns is being called with two None options. e.g. Neither a path to a"
            " file of columns or list of columns to keep has been provided to the function."
        )
        raise ValueError(error_msg)

    if columns_path:
        columns = get_subset_columns(columns_path)
    else:
        # Garuanteed to not be Null
        columns = columns_keep  # type: ignore
    sample_col = profiles.columns[0]
    profile_cols = set(profiles.columns[1:])  # keep all columns but first
    columns_subset = list(profile_cols & columns)
    return profiles.select(sample_col, *columns_subset)


def get_subset_columns(columns_path: Path) -> set[str]:
    """Load the columns used for subsetting loci."""
    with columns_path.open("r") as columns_file:
        columns: set[str] = {line.strip() for line in columns_file.readlines()}
    return columns


def verify_dataframe_integrity(profiles: pl.DataFrame) -> None:
    """Perform sanity checks on dataframes shape."""
    if profiles.shape[1] <= 1:
        err_string = (
            f"Sorry, {profiles.shape[1]} allele loci column(s) provied as input, but atleast two "
            f"loci columns are needed."
        )

        logger.critical(err_string)
        raise pl.exceptions.ShapeError(err_string)

    if profiles.shape[0] <= 1:
        err_string = (
            f"Sorry, {profiles.shape[0]} allele loci profile(s) provided as input, but atleast two "
            f"loci profiles are needed."
        )
        logger.critical(err_string)
        raise pl.exceptions.RowsError(err_string)

    if profiles.select(pl.nth(0).null_count())[0, 0] >= 1:
        err_string = (
            "Sorry, missing values identified in left most column (ID column). The left most "
            "column can have no missing values."
        )
        logger.critical(err_string)
        raise MissingIDValueError(err_string)

    if not profiles.select(pl.nth(0)).is_unique().all():
        err_string = (
            "Sorry, duplicate values identified in the left most column (ID column). The leftmost "
            "column must have no missing values."
        )
        logger.critical(err_string)
        raise pl.exceptions.DuplicateError(err_string)


def read_input_profiles(input_file: Path, delimiter: str, threads: int) -> pl.DataFrame:
    """Read the allelic profiles and perform any filtering and conversions required.

    Check for nulls in sample id column and enforce uniqueness.

    Polars mangles columns with potential duplicate names on loading, potential bug*
    """
    profiles = pl.read_csv(
        input_file,
        separator=delimiter,
        n_threads=threads,
        has_header=True,
        raise_if_empty=True,
        infer_schema=False,
    )

    profiles = profiles.with_columns(
        pl.all().exclude(profiles.columns[0]).replace(old=list(REPLACE_CHARS.keys()), new=None)
    )
    # Remove rows which are all empty e.g. caused by new lines at the end of files
    profiles = profiles.filter(~pl.all_horizontal(pl.all().is_null()))

    profiles = profiles.with_columns(
        pl.all()
        .exclude(profiles.columns[0])
        .cast(pl.Categorical, strict=False)  # strict is false otherwise null is an error
    )

    verify_dataframe_integrity(profiles)

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
        "Setting filter threshold to excluded rows missing %s or more loci.",
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


    Note: This function requires the data in polars to not be read in as categorical but as strings
    and no nulls present in the database.
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


def transform_data_categorical_encoding(profiles: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Return data prepared for calc_dists.

    Transform the DataFrame of profiles by creating a look up table to cast values to integers,
    converting missing allele charactars to zeroes and filtering rows.

    Uses unpivot which works but we have observed slow downs on large datasets and
    segmentation faults.
    """
    profiles = profiles.with_columns(
        pl.all().exclude(profiles.columns[0]).to_physical().cast(pl.UInt32)
    )

    """
    Replace 0 with some unique value, we do not need to re-encode every value just the value that
    is currently 0. Zero will then be used to encode null values.
    """
    profiles = profiles.with_columns(
        pl.all().exclude(profiles.columns[0]).replace(0, pl.UInt32.max())
    )
    profiles = profiles.fill_null(0)
    profiles = filter_rows(profiles, threshold)
    return profiles


def prep_data(
    profiles: pl.DataFrame,
    threshold: float,
    transformation_func: Callable[[pl.DataFrame, float], pl.DataFrame],
) -> npt.NDArray:
    """Prepare profiles for computation using the the calc_dists function of beave."""
    data_columns = profiles.columns[1:]  # only apply functions to loci columns
    profiles = transformation_func(profiles, threshold)
    profiles_numpy = (
        profiles.select(pl.col(data_columns)).to_numpy(writable=False, order="c").astype(np.uint32)
    )
    return profiles_numpy
