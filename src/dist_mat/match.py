"""Module for fast-matching process."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy.typing as npt
import polars as pl

import dist_mat.transform_data as transform
from dist_mat import fast_match
from dist_mat.log import init_logger

logger = init_logger(__name__)

MAX_ROWS_WRITE_BATCH: int = 1_000_000


class MatchColumns(StrEnum):
    """The final column names for the fast-matching output."""

    QUERY = "query_id"
    REFERENCE = "ref_id"


@dataclass(slots=True)
class MatchArguments:
    """CLI arguments for match process."""

    query: Path
    reference: Path
    threshold: float
    cores: int
    columns_path: Path | None
    delimiter: str
    count_missing: bool
    scaled: bool
    filter_threshold: float
    output: Path


class ColumnsDoNotMatchError(ValueError):
    """Exception raised for mismatching columns."""

    __max_print_value = 10

    def __init__(self, values: set[str]) -> None:
        """Error raised if columns do not match.

        The number of mismatching columns is printed, however if the number of
        mismatching columns is small they are printed to stderr. The threshold
        for printing columns to screen is determined by __max_print_value.

        The printing of columns to screen is restricted to prevent filling stdout
        with information that may obscure other useful log messages.
        """
        output_string: str = (
            f"The loci in your reference set do not match those in your query."
            f" Too many differences to list: {len(values)}"
        )
        if len(values) < self.__max_print_value:
            output_string = (
                f"The loci in your reference set do not match those "
                f"in your query.\n{'\n-'.join(values)}"
            )
        super().__init__(output_string)


def merge_query_and_reference(query: pl.DataFrame, reference: pl.DataFrame) -> pl.DataFrame:
    """Concatenate the query and reference profiles."""
    return pl.concat([query, reference], how="vertical")


def run_fast_matching(
    profiles: npt.NDArray,
    query_size: int,
    match_args: MatchArguments,
) -> npt.NDArray:
    """Generate fast-match results of query vs reference samples."""
    fast_match_data: npt.NDArray = fast_match(
        profiles,
        match_args.cores,
        match_args.scaled,
        match_args.count_missing,
        query_size,
        match_args.threshold,
    )
    return fast_match_data


def prepare_slice_to_write(
    slice: npt.NDArray, schema: pl.Schema, id_columns: npt.NDArray
) -> pl.DataFrame:
    """Prepare polars DataFrame of the final outputs for writing to a csv."""
    output_data = pl.from_numpy(
        slice,
        schema=schema,
    )

    output_data = output_data.with_columns(
        pl.col([MatchColumns.QUERY, MatchColumns.REFERENCE]).map_elements(
            lambda x: id_columns[int(x)]
        )
    )
    return output_data


def prepare_fast_match_outputs(
    data: npt.NDArray,
    profiles: npt.NDArray,
    match_args: MatchArguments,
) -> None:
    """Write out fast-match results for each query and reference."""
    dist_type: str = transform.DistanceTypes.HAMMING
    type_conversion: type[pl.UInt32] | type[pl.Float32] = pl.UInt32
    if match_args.scaled:
        dist_type = transform.DistanceTypes.SCALED
        type_conversion = pl.Float32
    output_schema = pl.Schema(
        {
            MatchColumns.QUERY: pl.UInt32,
            MatchColumns.REFERENCE: pl.UInt32,
            f"dist_{dist_type}": type_conversion,
        }
    )

    output_data: pl.DataFrame = prepare_slice_to_write(
        data[:MAX_ROWS_WRITE_BATCH], output_schema, profiles
    )
    logger.info(f"Writing to {match_args.output}.")
    output_data.write_csv(match_args.output, separator=match_args.delimiter)

    if len(data) < MAX_ROWS_WRITE_BATCH:
        """
        If the length of data is less than max_int, we can exit the program now.
        However list slicing is not inclusive of the final index, therefore we must
        still proceed to an additional write, even if the number of values is
        equal to max_int
        """
        return None

    logger.info(f"Final output is being written in batches of {MAX_ROWS_WRITE_BATCH}.")
    # write additional outputs if a 32 bit integer is exceeded
    with open(match_args.output, "a") as output:
        for idx in range(MAX_ROWS_WRITE_BATCH, len(data), MAX_ROWS_WRITE_BATCH):
            logger.debug(f"Writing batch {idx}-{idx + MAX_ROWS_WRITE_BATCH}")
            output_data = prepare_slice_to_write(
                data[idx : idx + MAX_ROWS_WRITE_BATCH], output_schema, profiles
            )
            # Do not print header as
            output_data.write_csv(output, include_header=False, separator=match_args.delimiter)


def match(match_args: MatchArguments) -> None:
    """Driver function for fast-matching."""
    logger.debug("Launching fast-matching.")
    with pl.StringCache():
        query = transform.read_input_profiles(
            match_args.query, match_args.delimiter, match_args.cores
        )
        logger.debug("Finished reading query profiles.")
        reference = transform.read_input_profiles(
            match_args.reference, match_args.delimiter, match_args.cores
        )
        logger.debug("Finished reading reference profiles.")

    logger.info("Finished reading reference and query profiles.")
    if match_args.columns_path:
        columns_to_keep = transform.get_subset_columns(match_args.columns_path)
        logger.info("Loaded columns to subset from profiles.")
        query = transform.subset_columns(query, None, columns_to_keep)
        logger.debug("Subset reference and query columns.")
        reference = transform.subset_columns(reference, None, columns_to_keep)
        logger.debug("Subset reference columns.")
        logger.info("Subset reference and query columns.")

    # Offset columns list by 1 to ignore the index column
    reference_columns = set(reference.columns[1:])
    query_columns = set(query.columns[1:])
    if reference_columns != query_columns:
        diff = reference_columns ^ query_columns  # report symmetric diff
        raise ColumnsDoNotMatchError(diff)

    merged_profiles = merge_query_and_reference(query, reference)
    logger.info("Merged query and reference profiles.")
    transform.verify_dataframe_integrity(merged_profiles)
    logger.debug("Finished verifying merged profiles dataframe.")

    profiles_prepared: npt.NDArray = transform.prep_data(
        merged_profiles, match_args.filter_threshold, transform.transform_data_categorical_encoding
    )
    logger.debug("Converted prepared profiles to numpy array.")

    fast_match_results: npt.NDArray = run_fast_matching(profiles_prepared, query.height, match_args)
    logger.info(f"Finished calculations and writing to output: {match_args.output}")

    """
    Need to provide an index row to the passed labels or else the look up of each value from
    the list when writing the output is incredibly slow.
    """
    samples: npt.NDArray = merged_profiles.select(pl.first()).to_series().to_numpy()
    prepare_fast_match_outputs(fast_match_results, samples, match_args)
    logger.info("Finished.")
