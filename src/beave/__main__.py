"""Main entry point for dist-mat.

This module contains the main cli for dist-mat.
"""

import importlib.metadata

__description__ = importlib.metadata.metadata(__package__ or __name__)["Summary"]
__version__ = importlib.metadata.version(__package__ or __name__)

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from beave import log
from beave.cluster import (
    BranchType,
    ClusterArguments,
    LinkageMetric,
    cluster,
)
from beave.match import MatchArguments, match

logger = log.init_logger(__name__)

MAX_PERCENT: float = 100.0


class Infinity(float):
    """Override of the float class to change the __repr__ of inifinity."""

    def __new__(cls) -> "Infinity":
        """Override instance creation of float class."""
        return super().__new__(cls, "infinity")

    def __repr__(self) -> str:
        """Override of repr for floats, to only return infinity."""
        return "infinity"


INFINITY: Infinity = Infinity()


class ArgValidator:
    """Class for validating CLI arguments based on passed functions.

    None of the passed arguments in this class peform any type conversion,
    but should instead raise an error if their passes arguments fail validation.
    """

    def __init__(self, /, validation_functions: Sequence[Callable[[Any], None]] = ()) -> None:
        """Intializer function for registered arguments."""
        self.validation_functions: Sequence[Callable[[Any], None]] = validation_functions

    def add_validation_function(
        self, new_function: Callable[[Any], None], *, prepend: bool = False
    ) -> None:
        """Add new validation function to class."""
        if not self.validation_functions:
            self.validation_functions = (new_function,)
        if prepend:
            self._prepend_validation_function(new_function)
        else:
            self._add_validation_function(new_function)

    def _add_validation_function(self, new_function: Callable[[Any], None]) -> None:
        """Add an additional validation function to the passed methods."""
        self.validation_functions = (*self.validation_functions, new_function)

    def _prepend_validation_function(self, new_function: Callable[[Any], None]) -> None:
        """Add an additional validation function to the front of the passed methods."""
        self.validation_functions = (new_function, *self.validation_functions)

    def __call__(self, argument: Any) -> None:
        """Apply validation methods to passed arguments."""
        if self.validation_functions is None:
            return

        for func in self.validation_functions:
            func(argument)


class VerboseLogAction(argparse.Action):
    """Custom action class that supports changing log verbosity."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        default: bool = False,
        required: bool = False,
        help: str | None = None,
    ) -> None:
        """Use parent class intializer."""
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(  # pyright: ignore[reportUnusedVariable]
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Override argparse action store_true and drop debug filter from logger stream handler."""
        log.SHARED_STREAM_HANDLER.removeFilter(log.DEBUG_FILTER)
        setattr(namespace, self.dest, self.const)


class CommandError(ValueError):
    """Generic value error for erroneous parameter entries."""

    def __init__(self, error: str) -> None:
        """Propogate string to value error for display to users."""
        super().__init__(error)


class Commands(StrEnum):
    """Sub-commands for the program."""

    CLUSTER = "cluster"
    MATCH = "match"


def path_exists(file_path: str) -> Path:
    """Check if path exists."""
    fp = Path(file_path)
    if fp.is_file():
        return fp
    error_message = f"Sorry, input file does not exist. {file_path}"
    logger.critical(error_message)
    raise FileNotFoundError(error_message)


def check_if_float(float_input: str) -> float:
    """Check if input value is float."""
    try:
        converted_float: float = float(float_input)
    except ValueError:
        error_message = f"Sorry, value  {float_input} cannot be converted to a float."
        logger.critical(error_message)
        raise ValueError(error_message)
    return converted_float


def percentage_range(float_input: str) -> float:
    """Check if input value is in range for comparisons."""
    converted_float: float = check_if_float(float_input)
    if converted_float <= 0.00 or converted_float > MAX_PERCENT:
        error_message = (
            f"Sorry, Filter threshold must be between 0.00 and 100.0. You passed: {float_input}"
        )
        logger.critical(error_message)
        raise ValueError(error_message)
    return converted_float / MAX_PERCENT  # convert percentage to decimal fraction


def cluster_threshold(float_input: str) -> float:
    """Verify input types are valid."""
    converted_input: float = float(float_input)
    if converted_input < 0.00 or converted_input == INFINITY:
        error_message = (
            f"Sorry, threshold values must be positive and not infinity. You passed: {float_input}"
        )
        logger.critical(error_message)
        raise ValueError(error_message)
    return converted_input


def fast_match_threshold(float_input: str) -> float:
    """Verify input types are valid."""
    converted_input: float = float(float_input)
    if converted_input < 0.00:
        error_message = f"Sorry, threshold values must be positive. You passed: {float_input}"
        logger.critical(error_message)
        raise ValueError(error_message)
    return converted_input


def verify_does_not_contain_infinity(thresholds: list[float]) -> None:
    """Verify input thresholds do not contain inifinity.

    This check is redundant, but adding direct logic check to simplify interface
    for later validation functions.
    """
    if INFINITY in thresholds:
        err_msg = f"Sorry, {INFINITY} can not be used as a threshold."
        logger.critical(err_msg)
        raise CommandError(err_msg)


def verify_normalized_distance(thresholds: float | list[float]) -> None:
    """Verify normalized distance thresholds."""
    max_value: float = max(thresholds) if isinstance(thresholds, list) else thresholds
    min_value: float = min(thresholds) if isinstance(thresholds, list) else thresholds
    max_value_bound_exceeded: bool = max_value != INFINITY and max_value >= MAX_PERCENT
    min_value_bound_exceeded: bool = min_value <= 0.0

    if not max_value_bound_exceeded and not min_value_bound_exceeded:
        return

    err_msg_max: str = (
        f"Sorry, normalized distance specified, but values greater than or equal to {MAX_PERCENT}"
        f" are provided. {max_value}"
    )
    err_msg_min: str = (
        f"Sorry, normalized distance specified, but values less than or equal to 0.0"
        f" are provided. {min_value}"
    )

    err_msg = []
    if max_value_bound_exceeded:
        err_msg.append(err_msg_max)
    if min_value_bound_exceeded:
        err_msg.append(err_msg_min)

    logger.critical("\n".join(err_msg))
    raise CommandError("\n".join(err_msg))


def output_directory(output: str) -> Path:
    """Create directory for output files."""
    handle: Path = Path(output)
    if not handle.is_dir():
        logger.debug("Creating output directory structure.")
        handle.mkdir(parents=True, exist_ok=True)
    return handle


def add_cluster_parser(parser_cluster: argparse.ArgumentParser) -> None:
    """Add the cluster parsing options to the parent parser."""
    # cluster args

    parser_cluster.add_argument(
        "--input", "-i", help="Input alleles.", type=path_exists, required=True
    )

    parser_cluster.add_argument(
        "--output",
        "-o",
        type=output_directory,
        required=False,
        help=(
            "Output directory for generated tree and clusters, directory will be treated if does"
            " not exist. (default: %(default)s)"
        ),
        default=os.getcwd(),
    )

    parser_cluster.add_argument(
        "--thresholds",
        "-t",
        help="List of threshold values to use.",
        nargs="+",
        required=True,
        action="extend",
        type=cluster_threshold,
    )

    parser_cluster.add_argument(
        "--linkage-method",
        "-l",
        default=LinkageMetric.AVERAGE.value,
        help="Hierarchical clustering linkage to use. (default: %(default)s)",
        choices=[i.value for i in LinkageMetric],
    )

    parser_cluster.add_argument(
        "--branch-type",
        "-b",
        default=BranchType.COPHENETIC.value,
        choices=[i.value for i in BranchType],
        help="Determine how to display tree lenghts in the Newick file. (default %(default)s)",
    )

    parser_cluster.add_argument(
        "--matrix", action="store_true", help="Write the computed distance matrix to a file."
    )


def add_match_parser(parser_match: argparse.ArgumentParser) -> None:
    """Add arguments to sub-parser for match."""
    parser_match.add_argument(
        "--reference",
        "-r",
        type=path_exists,
        required=True,
        help="Profiles to compare against. Query samples will be included in comparisons.",
    )

    parser_match.add_argument(
        "--query",
        "-q",
        type=path_exists,
        required=True,
        help="Profiles containing new-samples for comparisons.",
    )

    parser_match.add_argument(
        "--threshold",
        "-t",
        type=fast_match_threshold,
        help="Only report distances below specified threshold. (default: %(default)s)",
        default=INFINITY,
    )

    parser_match.add_argument(
        "--output",
        "-o",
        type=output_directory,
        required=False,
        help=("Output directory for calculated distances. (default: %(default)s)"),
        default=os.getcwd(),
    )


def create_parent_parser() -> argparse.ArgumentParser:
    """Create the parent parser for program."""
    parent_parser = argparse.ArgumentParser(  # Global command-line arguments:
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if cpu_count := os.cpu_count():
        number_of_cores_default = cpu_count // 2
    else:
        number_of_cores_default = 1

    parent_parser.add_argument(
        "--cores",
        "-c",
        help="Specify the number of threads to be used. (default %(default)d)",
        type=int,
        default=number_of_cores_default,
    )
    parent_parser.add_argument(
        "--delimiter",
        help="Input alleles delimiter. (default \\t)",
        type=str,
        default="\t",
    )

    parent_parser.add_argument(
        "--columns-subset",
        "-s",
        help=(
            "A file containing a single column of the column names to subset from the passed "
            "allele profiles."
        ),
        type=Path,
        required=False,
    )

    parent_parser.add_argument(
        "--count-missing",
        "-m",
        help="Count missing values as differences.",
        action="store_true",
    )

    parent_parser.add_argument(
        "--normalize-distance",
        "-n",
        help=(
            "Compute the normalized distance. Distance is presented as a percentage, or a value "
            "between [0.0-100.0]"
        ),
        action="store_true",
    )

    parent_parser.add_argument(
        "--filter-threshold",
        "-f",
        help=(
            "Exclude samples from analysis if they are missing more than the specified percentage "
            "of data. Must be between [0.0-100.0]. (default 100.0)"
        ),
        default=percentage_range("100.00"),
        type=percentage_range,
    )

    parent_parser.add_argument(
        "--verbose", action=VerboseLogAction, help="Display logger debug messages."
    )

    parser = argparse.ArgumentParser(
        description=__description__,
        parents=[parent_parser],
        allow_abbrev=True,
    )
    # suggest_on_error only exists in newer python versions, setting it as a @property
    # will not raise errors as the flag will just be un-used
    parser.suggest_on_error = True  # pyright: ignore[reportAttributeAccessIssue]

    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(
        help="Select a program to run.",
        dest="command",
    )

    parser_match = subparsers.add_parser(
        Commands.MATCH, help="Run fast matching.", parents=[parent_parser]
    )
    parser_cluster = subparsers.add_parser(
        Commands.CLUSTER, help="Run denovo clustering.", parents=[parent_parser]
    )

    add_cluster_parser(parser_cluster)
    add_match_parser(parser_match)

    return parser


async def main() -> None:
    """Program entry-point."""
    parser = create_parent_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit()

    validate_normalized_distance: ArgValidator = ArgValidator()
    if args.normalize_distance:
        validate_normalized_distance.add_validation_function(verify_normalized_distance)

    log.add_file_logger(args.output)
    match args.command:
        case Commands.CLUSTER:
            validate_normalized_distance.add_validation_function(
                verify_does_not_contain_infinity,
                prepend=True,  # Check can be used on hamming values as infinity is not allowed.
            )
            validate_normalized_distance(args.thresholds)
            cluster_args = ClusterArguments(
                args.input,
                args.delimiter,
                args.thresholds,
                args.linkage_method,
                args.cores,
                args.columns_subset,
                args.count_missing,
                args.normalize_distance,
                args.branch_type,
                args.filter_threshold,
                args.matrix,
                args.output,
            )
            await cluster(cluster_args)
        case Commands.MATCH:
            validate_normalized_distance(args.threshold)
            output_file = args.output / "results.tsv"
            match_args = MatchArguments(
                args.query,
                args.reference,
                args.threshold,
                args.cores,
                args.columns_subset,
                args.delimiter,
                args.count_missing,
                args.normalize_distance,
                args.filter_threshold,
                output_file,
            )
            match(match_args)
        case _:
            parser.print_help()
            sys.exit()


def async_main():
    """Async entry point for main python function."""
    asyncio.run(main())
