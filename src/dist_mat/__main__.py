import argparse
import sys
import os
import pathlib as p
from enum import StrEnum

from dist_mat import mcluster


class Commands(StrEnum):
    MCLUSTER = "mcluster"


class DistanceMetrics(StrEnum):
    WARD = "ward"
    SINGLE = "single"
    AVERAGE = "average"
    CENTROID = "centroid"
    MEDIAN = "median"
    COMPLETE = "complete"


def main():
    # specify global arguments shared here
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
        "Example program",
        description="A quick proof of concept of generic utilities for nomenclature assignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parent_parser],
    )

    parser.add_argument(
        "--version", "-v", help="Print version and exit.", action="store_true"
    )

    subparsers = parser.add_subparsers(
        help="Select a program to run.",
        dest="command",
    )

    # Mcluster args
    parser_mcluster = subparsers.add_parser(
        Commands.MCLUSTER, help="Run denovo clustering.", parents=[parent_parser]
    )

    parser_mcluster.add_argument(
        "--input", "-i", help="Input alleles.", type=p.Path, required=True
    )

    parser_mcluster.add_argument(
        "--tree-output",
        "-t",
        help="Output tree name. [default %(default)s]",
        type=p.Path,
        required=False,
        default="clusters.nwk",
    )

    parser_mcluster.add_argument(
        "--cluster-output",
        "-l",
        help="Output clusters file. [default %(default)s]",
        type=p.Path,
        required=False,
        default="clusters.tsv",
    )

    parser_mcluster.add_argument(
        "--thresholds",
        "-p",
        help="List of threshold values to use, can be Float or Int",
        nargs="+",
        required=True,
        action="extend",
        type=float,
    )

    parser_mcluster.add_argument(
        "--method",
        "-m",
        default=DistanceMetrics.AVERAGE.value,
        help="Linkage method to use. [default: %(default)s]",
        choices=[i.value for i in DistanceMetrics],
    )

    parser_mcluster.add_argument(
        "--columns",
        "-k",
        help="A file containing a list of columns to subset from the allele profiles.",
        type=p.Path,
        required=False,
    )

    parser_mcluster.add_argument(
        "--count-missing",
        "-c",
        help="Count missing values as differences.",
        action="store_true",
    )

    parser_mcluster.add_argument(
        "--scaled",
        "-s",
        help="Compute the scaled distance.",
        action="store_true",
    )

    args = parser.parse_args(sys.argv[1:])

    if args.version:
        print("0.0.1")
        sys.exit()

    match args.command:
        case Commands.MCLUSTER:
            mcluster.mcluster(
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
            )
        case _:
            parser.print_help()
            sys.exit()


if __name__ == "__main__":
    main()
