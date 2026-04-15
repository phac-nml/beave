"""Internal functions used for data loading and transformation."""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from dist_mat._internal.log import init_logger

logger = init_logger(__name__)


class AllColumnsFilteredError(Exception):
    """Exception raised if all rows are filtered."""

    def __init__(self, threshold: float) -> None:
        """Error raised if all column values removed."""
        super().__init__(f"All data removed after filtering at: {threshold}.")


MISSING_VALUE = np.uint32(0)
REPLACE_CHARS = {
    "?": MISSING_VALUE,
    " ": MISSING_VALUE,
    "-": MISSING_VALUE,
    "": MISSING_VALUE,
    "_": MISSING_VALUE,
    "0": MISSING_VALUE,
}  # mappings to replace fields with zeroes


def subset_columns(
    profiles: pl.DataFrame, /, columns_path: Path | None, columns_keep: set[str] | None
) -> pl.DataFrame:
    """Subset only the required columns to use for distance matrix calculation."""
    columns: set[str]

    if columns_path is None and columns_keep is None:
        error_msg = "Subset columns is being called with two None options."
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


def verify_dataframe_integrity(profiles: pl.DataFrame):
    """Perform sanity checks on dataframes shape."""
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


def read_input_profiles(input_file: Path, delimiter: str, threads: int) -> pl.DataFrame:
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
    # Remove rows which are all empty e.g. caused by new lines at the end of files
    profiles = profiles.filter(~pl.all_horizontal(pl.all() == ""))
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
        "Setting filter threshold to excluded columns missing %s or more loci.",
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
        profiles.select(pl.all().exclude(profiles.columns[0]))
        .unpivot()
        .to_series(values_columns)
        .unique()
        .to_list()
    )
    logger.debug("Identified unique values for re-mapping.")

    char_mapping = (
        {  # start mapping at 1, as 0 is used for missing values and add one to not miss values
            value: idx
            for value, idx in zip(
                unique_values, np.arange(1, len(unique_values) + 1, dtype=np.uint32)
            )
        }
        | REPLACE_CHARS
    )  # Create new dictionary, REPLACE_CHARS keys overwrite those in new dictionary

    logger.debug("Replacing profiles with integer mapping.")
    profiles = profiles.with_columns(
        pl.all()
        .exclude(profiles.columns[0])  # skip id column
        .replace(char_mapping)
        .cast(pl.UInt32)  # strict cast will throw an error if any overflow occurs
    )
    logger.debug("Finished replacing profiles with integer mapping.")

    profiles = filter_rows(profiles, threshold)
    return profiles


def prep_data(
    profiles: pl.DataFrame,
    threshold: float,
    transformation_func: Callable[[pl.DataFrame, float], pl.DataFrame],
) -> npt.NDArray:
    """Prepare profiles for computation by the the calc_dists function of dist_mat."""
    data_columns = profiles.columns[1:]  # only apply functions to loci columns
    profiles = transformation_func(profiles, threshold)
    profiles_numpy = (
        profiles.select([pl.col(i) for i in data_columns])
        .to_numpy(writable=False)
        .astype(np.uint32)
    )
    return profiles_numpy
