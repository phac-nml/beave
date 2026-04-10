"""Module for fast-matching process."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import dist_mat._internal.transform_data as transform
from dist_mat import fast_match
from dist_mat._internal.log import init_logger

logger = init_logger(__name__)


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


def prepare_fast_match_outputs(
    data: npt.NDArray, profiles: pl.DataFrame, match_args: MatchArguments
) -> None:
    """Write out fast-match results for each query and reference."""
    with match_args.output.open("w") as dists_out:
        dist_type: str = "hamming"
        type_conversion: Callable[[np.float32], np.uint32] | Callable[[np.float32], np.float32] = (
            np.uint32
        )
        if match_args.scaled:
            dist_type = "scaled"
            type_conversion = np.float32
        print("query_id", "ref_id", f"dist_{dist_type}", sep="\t", file=dists_out)
        for row in data:
            print(
                profiles.row(int(row[0]))[1],  # 1 gets the valule offset from the index
                profiles.row(int(row[1]))[1],
                type_conversion(row[2]),
                sep="\t",
                file=dists_out,
            )


def match(match_args: MatchArguments) -> None:
    """Driver function for fast-matching."""
    query = transform.read_input_profiles(match_args.query, match_args.delimiter, match_args.cores)
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
    reference_columns = set(reference.columns)
    query_columns = set(query.columns)
    if reference_columns != query_columns:
        diff = reference_columns ^ query_columns  # report symmetric diff
        raise ColumnsDoNotMatchError(diff)

    merged_profiles = merge_query_and_reference(query, reference)
    logger.info("Merged query and reference profiles.")
    transform.verify_dataframe_integrity(merged_profiles)
    logger.debug("Finished verifying merged profiles dataframe.")

    profiles_prepared: npt.NDArray = transform.prep_data(
        merged_profiles, match_args.filter_threshold, transform.transform_data
    )
    logger.debug("Converted prepared profiles to numpy array.")

    fast_match_results: npt.NDArray = run_fast_matching(profiles_prepared, query.height, match_args)
    logger.info(f"Finished calculations and writing to output: {match_args.output}")

    """
    Need to provide an index row to the passed labels or else the look up of each value from
    the list when writing the output is incredibly slow.
    """
    samples: pl.DataFrame = merged_profiles.select(pl.first()).with_row_index()
    prepare_fast_match_outputs(fast_match_results, samples, match_args)
    logger.info("Finished.")
