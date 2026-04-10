"""Main entry point for dist-mat.

This module contains the main cli for dist-mat.
"""

import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

import argparse
import logging
import os
import sys
from enum import StrEnum
from pathlib import Path

from dist_mat._internal.log import init_logger
from dist_mat.cluster import (
    BranchLengthType,
    ClusterArguments,
    LinkageMetric,
    cluster,
)
from dist_mat.match import MatchArguments, match

logger = init_logger(__name__)

MAX_PERCENT: float = 100.0


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
    error_message = f"Input file does not exist. {file_path}"
    logger.critical(error_message)
    raise FileNotFoundError(error_message)


def check_if_float(float_input: str) -> float:
    """Check if input value is float."""
    try:
        converted_float: float = float(float_input)
    except ValueError:
        error_message = f"Value  {float_input} cannot be converted to a float."
        logger.critical(error_message)
        raise ValueError(error_message)
    return converted_float


def percentage_range(float_input: str) -> float:
    """Check if input value is in range for comparisons."""
    converted_float: float = check_if_float(float_input)
    if converted_float < 0.00 or converted_float > MAX_PERCENT:
        error_message = (
            f"Filter threshold must be between 0.00 and 100.0. You passed: {float_input}"
        )
        logger.critical(error_message)
        raise ValueError(error_message)
    return converted_float / MAX_PERCENT  # convert percentage to decimal fraction


def cluster_threshold(float_input: str) -> float:
    """Verify input types are valid."""
    converted_input: float = float(float_input)
    if converted_input < 0.00 or converted_input == float("inf"):
        error_message = (
            f"Threshold values must be positive and not infinity. You passed: {float_input}"
        )
        logger.critical(error_message)
        raise ValueError(error_message)
    return converted_input


def verify_scaled_distance(scaled: bool, thresholds: float | list[float]) -> None:
    """Verify scaled distance thresholds."""
    if not scaled:
        return

    test_value: float = max(thresholds) if isinstance(thresholds, list) else thresholds
    if test_value == float("inf") or test_value <= MAX_PERCENT:
        return

    err_msg: str = "Scaled distance specified, but values greater than 100.0 are specified."
    logger.critical(err_msg)
    raise CommandError(err_msg)


def output_file(output: str) -> Path:
    """Create directory for output results file if needed."""
    handle: Path = Path(output)
    if not handle.parent.is_dir():
        logger.debug("Creating output directory structure.")
        handle.parent.mkdir(parents=True, exist_ok=True)
    return handle


def main() -> None:
    """Program entry-point."""
    parent_parser = argparse.ArgumentParser(  # Global command-line arguments:
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if cpu_count := os.cpu_count():
        number_of_cores_default = cpu_count // 2
    else:
        number_of_cores_default = 1

    parent_parser.add_argument(
        "--n-threads",
        "-n",
        help="Specify the number of threads to be used. [default %(default)d]",
        type=int,
        default=number_of_cores_default,
    )
    parent_parser.add_argument(
        "--delimiter",
        "-d",
        help="Input alleles delimiter. [default \\t]",
        type=str,
        default="\t",
    )

    parent_parser.add_argument(
        "--columns",
        "-k",
        help=(
            "A file containing a single column of the column names to subset from the passed "
            "allele profiles."
        ),
        type=Path,
        required=False,
    )

    parent_parser.add_argument(
        "--count-missing",
        "-c",
        help="Count missing values in allele profiles differences.",
        action="store_true",
    )

    parent_parser.add_argument(
        "--scaled",
        "-s",
        help=(
            "Compute the scaled distance. Distance is presented as a percentage, or a value "
            "between 0.0-100.0"
        ),
        action="store_true",
    )

    parent_parser.add_argument(
        "--filter-threshold",
        "-f",
        help=(
            "Excluded samples from analysis if it is missing more than the specified percentage "
            "of data. Must be between 0.0 and 100.0. [default %(default)s]"
        ),
        default=100.00,
        type=percentage_range,
    )

    parent_parser.add_argument(
        "--verbose", action="store_true", help="Display logger debug messages."
    )

    parser = argparse.ArgumentParser(
        description="A quick proof of concept of generic utilities for nomenclature assignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parent_parser],
    )

    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(
        help="Select a program to run.",
        dest="command",
    )

    # cluster args
    parser_cluster = subparsers.add_parser(
        Commands.CLUSTER, help="Run denovo clustering.", parents=[parent_parser]
    )

    parser_cluster.add_argument(
        "--input", "-i", help="Input alleles.", type=path_exists, required=True
    )

    parser_cluster.add_argument(
        "--tree-output",
        "-t",
        help="File path to write generated tree. [default %(default)s]",
        type=output_file,
        required=False,
        default="clusters.nwk",
    )

    parser_cluster.add_argument(
        "--cluster-output",
        "-l",
        help="File path to write generated clusters. [default %(default)s]",
        type=output_file,
        required=False,
        default="clusters.tsv",
    )

    parser_cluster.add_argument(
        "--thresholds",
        "-p",
        help="List of threshold values to use.",
        nargs="+",
        required=True,
        action="extend",
        type=cluster_threshold,
    )

    parser_cluster.add_argument(
        "--method",
        "-m",
        default=LinkageMetric.AVERAGE.value,
        help="Hierarchical clustering linkage to use. [default: %(default)s]",
        choices=[i.value for i in LinkageMetric],
    )

    parser_cluster.add_argument(
        "--tree-distances",
        "-b",
        default=BranchLengthType.COPHENETIC.value,
        choices=[i.value for i in BranchLengthType],
        help="Determine how to display tree lenghts in the newick file. [default %(default)s]",
    )

    parser_match = subparsers.add_parser(
        Commands.MATCH, help="Run fast matching.", parents=[parent_parser]
    )

    parser_match.add_argument(
        "--reference", "-r", type=path_exists, required=True, help="Profiles to compare against."
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
        type=cluster_threshold,
        help="Only report distances below specified threshold. [default: %(default)s]",
        default=float("inf"),
    )

    parser_match.add_argument(
        "--output",
        "-o",
        type=Path,
        required=False,
        help="Fast match result output tsv file. [default: %(default)s]",
        default=output_file("output.tsv"),
    )

    args = parser.parse_args(sys.argv[1:])

    if args.verbose:
        """Set the root loggers level to debug if verbose is enabled."""
        logging.getLogger().setLevel(logging.DEBUG)

    match args.command:
        case Commands.CLUSTER:
            verify_scaled_distance(args.scaled, args.thresholds)
            cluster_args = ClusterArguments(
                args.input,
                args.delimiter,
                args.thresholds,
                args.method,
                args.n_threads,
                args.columns,
                args.count_missing,
                args.scaled,
                args.tree_output,
                args.cluster_output,
                args.tree_distances,
                args.filter_threshold,
            )
            cluster(cluster_args)
        case Commands.MATCH:
            verify_scaled_distance(args.scaled, args.threshold)
            match_args = MatchArguments(
                args.query,
                args.reference,
                args.threshold,
                args.n_threads,
                args.columns,
                args.delimiter,
                args.count_missing,
                args.scaled,
                args.filter_threshold,
                args.output,
            )
            match(match_args)
        case _:
            parser.print_help()
            sys.exit()
