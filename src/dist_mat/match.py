"""Module for fast-matching process."""

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


def prepare_fast_match_outputs(data: npt.NDArray, profiles: pl.DataFrame, output: Path) -> None:
    """Write out fast-match results for each query and reference."""
    with output.open("w") as dists_out:
        dists_out.write("query_id\tref_id\tdist")
        for row in data:
            print(
                profiles.row(int(row[0]))[0],
                profiles.row(int(row[1]))[0],
                np.float32(row[2]),
                sep="\t",
                file=dists_out,
            )


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

    profiles_prepared: npt.NDArray = transform.prep_data(
        merged_profiles, match_args.filter_threshold, transform.transform_data
    )

    fast_match_results: npt.NDArray = run_fast_matching(profiles_prepared, query.height, match_args)
    prepare_fast_match_outputs(fast_match_results, merged_profiles, match_args.output)
