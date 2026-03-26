"""Main entry point for dist-mat

This module contains the main cli for dist-mat.
"""

import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

import argparse
import sys
import os
import logging
import pathlib as p
from enum import StrEnum

from dist_mat.cluster import (
    cluster,
    LinkageMetric,
    BranchLengthType,
    ClusterArguments,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Commands(StrEnum):
    CLUSTER = "cluster"


def path_exists(file_path: str) -> p.Path:
    fp = p.Path(file_path)
    if fp.is_file():
        return fp
    error_message = f"Input file does not exist. {file_path}"
    logger.critical(error_message)
    raise FileNotFoundError(error_message)


def check_if_float(float_input: str) -> float:
    try:
        coerced_float: float = float(float_input)
    except ValueError:
        error_message = f"Value  {float_input} cannot be coerced to a float."
        logger.critical(error_message)
        raise ValueError(error_message)
    return coerced_float


def percentage_range(float_input: str) -> float:
    coerced_float = check_if_float(float_input)
    if coerced_float < 0.00 or coerced_float > 100.0:
        error_message = (
            f"Filter threshold must be between 0.00 and 100.0. You passed: {float_input}"
        )
        logger.critical(error_message)
        raise ValueError(error_message)
    return coerced_float


def cluster_threshold(float_input: str) -> float:
    coerced_input: float = float(float_input)
    if coerced_input < 0.00 or coerced_input == float("inf"):
        error_message = (
            f"Threshold values must be positive and not infinity. You passed: {float_input}"
        )
        logger.critical(error_message)
        raise ValueError(error_message)
    return coerced_input


def main() -> None:
    # Global command-line arguments:
    parent_parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    number_of_cores_default = 1
    if cpu_count := os.cpu_count():
        number_of_cores_default = cpu_count // 2

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

    # Mcluster args
    parser_mcluster = subparsers.add_parser(
        Commands.MCLUSTER, help="Run denovo clustering.", parents=[parent_parser]
    )

    parser_mcluster.add_argument(
        "--input", "-i", help="Input alleles.", type=path_exists, required=True
    )

    parser_mcluster.add_argument(
        "--tree-output",
        "-t",
        help="File path to write generated tree. [default %(default)s]",
        type=p.Path,
        required=False,
        default="clusters.nwk",
    )

    parser_mcluster.add_argument(
        "--cluster-output",
        "-l",
        help="File path to write generated clusters. [default %(default)s]",
        type=p.Path,
        required=False,
        default="clusters.tsv",
    )

    parser_mcluster.add_argument(
        "--thresholds",
        "-p",
        help="List of threshold values to use.",
        nargs="+",
        required=True,
        action="extend",
        type=cluster_threshold,
    )

    parser_mcluster.add_argument(
        "--method",
        "-m",
        default=LinkageMetric.AVERAGE.value,
        help="Hierarchical clustering linkage to use. [default: %(default)s]",
        choices=[i.value for i in LinkageMetric],
    )

    parser_mcluster.add_argument(
        "--columns",
        "-k",
        help="A file containing a single column of the column names to subset from the passed allele profiles.",
        type=p.Path,
        required=False,
    )

    parser_mcluster.add_argument(
        "--count-missing",
        "-c",
        help="Count missing values in allele profiles differences.",
        action="store_true",
    )

    parser_mcluster.add_argument(
        "--scaled",
        "-s",
        help="Compute the scaled distance. Distance is presented as a percentage, or a value between 0.0-100.0",
        action="store_true",
    )

    parser_mcluster.add_argument(
        "--tree-distances",
        "-b",
        default=BranchLengthType.COPHENETIC.value,
        choices=[i.value for i in BranchLengthType],
        help="Determine how to display tree lenghts in the newick file. [default %(default)s]",
    )

    parser_mcluster.add_argument(
        "--filter-threshold",
        "-f",
        help="Excluded samples from analysis if it is missing more than the specified percentage of data. Must be between 0.0 and 100.0. [default %(default)s]",
        default=0.00,
        type=percentage_range,
    )

    args = parser.parse_args(sys.argv[1:])

    match args.command:
        case Commands.CLUSTER:
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
        case _:
            parser.print_help()
            sys.exit()
