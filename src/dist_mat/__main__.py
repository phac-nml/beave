import argparse
import sys
import os
import logging
import pathlib as p
from enum import StrEnum

from dist_mat.mcluster import mcluster, LinkageMetrics, BranchLengths

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Commands(StrEnum):
    MCLUSTER = "mcluster"


def path_exists(file: str) -> p.Path:
    fp = p.Path(file)
    if fp.is_file():
        return fp
    logger.critical(f"Input file does not exist. {file}")
    raise FileNotFoundError(f"Input file {file} does not exist.")


def percentage_range(f_input: str) -> float:
    try:
        coerced_input: float = float(f_input)
    except ValueError:
        logger.critical(f"Filter threshold  {f_input} cannot be coerced to a float.")
        # I do not know if this is the best way to bubble up a handled exception
        # but it allows me to raise the error without exiting directly and produce
        # a log message
        raise ValueError(f"Filter threshold {f_input} cannot be coerced to a float.")
    else:
        if coerced_input < 0.00 or coerced_input > 100.0:
            logger.critical(
                f"Filter threshold must be between 0.00 and 100.0. You passed: {f_input}"
            )
            raise ValueError(
                f"Filter threshold must be between 0.00 and 100.0. You passed: {f_input}"
            )
        return coerced_input


def main() -> None:
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
        "--input", "-i", help="Input alleles.", type=path_exists, required=True
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
        help="List of threshold values to use.",
        nargs="+",
        required=True,
        action="extend",
        type=float,
    )

    parser_mcluster.add_argument(
        "--method",
        "-m",
        default=LinkageMetrics.AVERAGE.value,
        help="Linkage method to use. [default: %(default)s]",
        choices=[i.value for i in LinkageMetrics],
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
        help="Compute the scaled distance. Distance is presented as a percentage, or a value between 0.0-100.0",
        action="store_true",
    )

    parser_mcluster.add_argument(
        "--tree-distances",
        "-b",
        default=BranchLengths.COPHENETIC.value,
        choices=[i.value for i in BranchLengths],
        help="Determine how to display tree lenghts in the newick file. [default %(default)s]",
    )

    parser_mcluster.add_argument(
        "--filter-threshold",
        "-f",
        help="Excluded samples from clustering missing more than a certain percentage of alleles must be between 0.0 and 100.0. [default %(default)s]",
        default=0.00,
        type=percentage_range,
    )

    args = parser.parse_args(sys.argv[1:])

    if args.version:
        print("0.0.1")
        sys.exit()

    match args.command:
        case Commands.MCLUSTER:
            mcluster(
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
        case _:
            parser.print_help()
            sys.exit()


if __name__ == "__main__":
    main()
