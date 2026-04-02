"""Module for fast-matching process."""

from dataclasses import dataclass
from pathlib import Path

import numpy.typing as npt
import polars as pl

import dist_mat._internal.transform_data as transform
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


class ColumnsDoNotMatchError(ValueError):
    """Exception raised for mismatching columns."""

    __max_print_value = 10

    def __init__(self, values: set[str]) -> None:
        """Error raised if all column values removed."""
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


def run_fast_matching(profiles: pl.DataFrame, threshold: float, query_length: int) -> npt.NDArray:
    """Generate fast-match results of query vs reference samples."""
    """
    Plan:
    Pass:
        - The profiles in array to C++ same as with clustering.
        - The threshold as a numpy float.
        - The index the terminal index of the last query sample.

    """
    ...
    raise NotImplementedError()


def match(match_args: MatchArguments) -> None:
    """Driver function for fast-matching."""
    query = transform.read_input_profiles(match_args.query, match_args.delimiter, match_args.cores)
    reference = transform.read_input_profiles(
        match_args.reference, match_args.delimiter, match_args.cores
    )
    if match_args.columns_path:
        columns_to_keep = transform.get_subset_columns(match_args.columns_path)
        query = transform.subset_columns(query, None, columns_to_keep)
        reference = transform.subset_columns(reference, None, columns_to_keep)

    # Offset columns list by 1 to ignore the index column
    reference_columns = set(reference.columns)
    query_columns = set(query.columns)
    if reference_columns != query_columns:
        diff = reference_columns ^ query_columns  # report symmetric diff
        raise ColumnsDoNotMatchError(diff)

    merged_profiles = merge_query_and_reference(query, reference)
    transform.verify_dataframe_integrity(merged_profiles)
