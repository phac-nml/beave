"""
Re-implementation of mcluster
"""

import sys
import logging
import pathlib as p


import dist_mat as dm
import scipy as sp
import polars as pl
import numpy as np
from numpy import typing as npt

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _scipy_tree_to_newick_list(node, newick, parentdist, leaf_names):
    """Construct Newick tree from SciPy hierarchical clustering ClusterNode

    This is a recursive function to help build a Newick output string from a scipy.cluster.hierarchy.to_tree input with
    user specified leaf node names.

    Notes:
        This function is meant to be used with `to_newick`

    Args:
        node (scipy.cluster.hierarchy.ClusterNode): Root node is output of scipy.cluster.hierarchy.to_tree from hierarchical clustering linkage matrix
        parentdist (float): Distance of parent node of `node`
        newick (list of string): Newick string output accumulator list which needs to be reversed and concatenated (i.e. `''.join(newick)`) for final output
        leaf_names (list of string): Leaf node names

    Returns:
        (list of string): Returns `newick` list of Newick output strings
    """
    if node.is_leaf():
        return newick + [f"{leaf_names[node.id]}:{parentdist - node.dist}"]

    if len(newick) > 0:
        newick.append(f"):{parentdist - node.dist}")
    else:
        newick.append(");")
    newick = _scipy_tree_to_newick_list(node.get_left(), newick, node.dist, leaf_names)
    newick.append(",")
    newick = _scipy_tree_to_newick_list(node.get_right(), newick, node.dist, leaf_names)
    newick.append("(")
    return newick


def to_newick(tree, leaf_names) -> str:
    """Newick tree output string from SciPy hierarchical clustering tree

    Convert a SciPy ClusterNode tree to a Newick format string.
    Use scipy.cluster.hierarchy.to_tree on a hierarchical clustering linkage matrix to create the root ClusterNode for the `tree` input of this function.

    Args:
        tree (scipy.cluster.hierarchy.ClusterNode): Output of scipy.cluster.hierarchy.to_tree from hierarchical clustering linkage matrix
        leaf_names (list of string): Leaf node names

    Returns:
        (string): Newick output string
    """
    newick_list = _scipy_tree_to_newick_list(tree, [], tree.dist, leaf_names)
    return "".join(newick_list[::-1])


def read_input_profiles(
    input: p.Path, columns_keep: p.Path | None, delimiter: str, threads: int
) -> pl.DataFrame:
    """
    Ingest the allelic profiles and perform any filtering and conversions required.
    """

    profiles = pl.read_csv(
        input, separator=delimiter, n_threads=threads, has_header=True
    )

    if columns_keep is not None:
        columns_to_keep: set[str] | None = None
        with columns_keep.open("r") as ck:
            columns_to_keep = set([line.strip() for line in ck.readlines()])

        sample_col = profiles.columns[0]
        profile_cols = set(profiles.columns[1:])  # keep all columns but first
        columns_keep_set = list(profile_cols & columns_to_keep)
        profiles = profiles.select(sample_col, *columns_keep_set)

    ## Can add additional filtering logic here
    return profiles


def prep_data(profiles: pl.DataFrame) -> npt.NDArray:
    """
    Prepare profiles for ingestion by the the calc_dists function of dist_mat.
    """

    data_columns = profiles.columns[1:]  # only apply functions to test columns
    missing_value = np.uint32(0)
    mapping_replace = {
        "?": missing_value,
        " ": missing_value,
        "-": missing_value,
        "": missing_value,
        "_": missing_value,
    }  # mappings to replace fields with zeroes
    profiles = profiles.with_columns(
        [pl.col(i).replace(mapping_replace) for i in data_columns]
    )

    profiles = profiles.with_columns(
        [
            pl.when(pl.col(i) != str(missing_value))
            .then(pl.col(i).hash(42, 42, 42, 42))
            .otherwise(pl.lit(0))
            for i in data_columns
        ]
    )

    profiles_numpy = profiles.select([pl.col(i) for i in data_columns]).to_numpy()
    profiles_numpy = profiles_numpy.astype(
        np.uint32
    )  # convert hashes to 32 bit representation for dist_mat
    return profiles_numpy


def compute_dists(
    profiles: pl.DataFrame, count_missing: bool, scaled: bool, threads: int
) -> npt.NDArray:
    """
    Compute the 1D array required by scipy for generation of the linkage matrix.
    """
    prepared_profiles = prep_data(profiles)
    distances = dm.calc_dists(prepared_profiles, threads, scaled, count_missing)
    return distances


def comp_linkage_matrix(
    profiles_computed: npt.NDArray, linkage_method: str
) -> npt.NDArray:
    linkage = sp.cluster.hierarchy.linkage(profiles_computed, method=linkage_method)
    return linkage


def assign_clusters(
    linkage: npt.NDArray, thresholds: list[int | float], labels: list[str]
):
    """
    Roll the new cluster membership code.

    Thresholds are assumed to be sorted on input
    """

    # cluster_members = pl.Schema(
    #    [("Sample", pl.String), *[(f"level_{str(k)}", pl.UInt32) for k in thresholds]]
    # )
    data_to_populate = [pl.Series(name="SampleID", values=labels)]
    cols_concat = []

    for threshold in thresholds:
        col_name = f"level_{str(threshold)}"
        data_to_populate.append(
            pl.Series(
                name=col_name,
                values=sp.cluster.hierarchy.fcluster(
                    linkage, threshold, criterion="distance"
                ),
                dtype=pl.UInt32,
            )
        )
        cols_concat.append(col_name)
    outputs = pl.DataFrame(data_to_populate, orient="col")
    outputs = outputs.with_columns(
        pl.concat_str(cols_concat, separator=".").alias("denovo_address")
    )
    return outputs


def mcluster(
    input: p.Path,
    delimiter: str,
    thresholds: list[float],
    methods: str,
    n_threads: int,
    columns: p.Path | None,
    count_missing: bool,
    scaled: bool,
    tree_output: p.Path,
    cluster_outputs: p.Path,
    *args,
    **kwargs,
):
    """
    Main runner function for mcluster.
    """

    profiles = read_input_profiles(input, columns, delimiter, n_threads)
    logger.info("Ingested profiles")
    distances = compute_dists(profiles, count_missing, scaled, n_threads)
    logger.info("Computed distances")
    linkages = comp_linkage_matrix(distances, methods)
    logger.info("Computed linkage matrix")
    thresholds.sort(reverse=True)
    logger.info(f"Thresholds being used for generating lingakes: {thresholds}")

    sample_names = profiles.select(pl.nth(0)).to_series().to_list()
    # write out the tree
    cluster_memberships = assign_clusters(linkages, thresholds, sample_names)
    logger.info("assigned clusters")

    sys.setrecursionlimit(4000)  # raise recursion limit for generating the tree
    tree = sp.cluster.hierarchy.to_tree(linkages)
    newick = to_newick(tree, sample_names)

    with tree_output.open("w") as to:
        to.write(newick)
    logger.info(f"Wrote newick tree to: {str(tree_output)}")

    cluster_memberships.write_csv(
        cluster_outputs, separator=delimiter, include_header=True
    )
    logger.info(f"Wrote cluster memberships to: {str(cluster_outputs)}")
