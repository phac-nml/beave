#! /usr/bin/env python3

import numpy as np

import polars as pd
import scipy as sp
import time
import sys
import dist_mat


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


prog_time = time.perf_counter()

start_time = time.perf_counter()
test_path = "/tmp/KData/merged/profile.tsv"
locidex_data = pd.read_csv(test_path, separator="\t", n_threads=4, has_header=True)
end_time = time.perf_counter()
print(f"Read table: {end_time - start_time:.4f} seconds.")

start_time = time.perf_counter()
locidex_data = locidex_data.with_columns(
    pd.nth([i for i in range(1, locidex_data.shape[1])])
)
locidex_data = locidex_data.with_columns(
    pd.nth([i for i in range(1, locidex_data.shape[1])]).hash(42, 42, 42, 42)
)
end_time = time.perf_counter()
ldx_nump = locidex_data.select(
    pd.nth([i for i in range(1, locidex_data.shape[1])])
).to_numpy()
ldx_nump = ldx_nump.astype(np.uint32)
# print(locidex_data)
# print(ldx_nump)
print(f"Processed input files: {end_time - start_time:.4f} seconds.")


start_time = time.perf_counter()
output_dists = dist_mat.calc_dists(ldx_nump, 16, True, True)
end_time = time.perf_counter()
print(f"Calculated distances: {end_time - start_time:.4f} seconds.")
# print(output_dists)

# print(f"Length of array passed to scipy.linkage: {len(output_dists)}")
start_time = time.perf_counter()
linkage = sp.cluster.hierarchy.linkage(output_dists, method="average")
end_time = time.perf_counter()
print(f"Performed linkage step: {end_time - start_time:.4f} seconds.")

start_time = time.perf_counter()
clusters = sp.cluster.hierarchy.fcluster(linkage, t=2.0, criterion="distance")
end_time = time.perf_counter()
print(f"Performed fcluster: {end_time - start_time:.4f} seconds.")

start_time = time.perf_counter()
sys.setrecursionlimit(4000)
T = sp.cluster.hierarchy.to_tree(linkage)
nwk = to_newick(T, locidex_data.select(pd.nth(0)).to_series().to_list())
with open("output_nwk.nwk", "w") as onwk:
    onwk.write(nwk)
end_time = time.perf_counter()
print(f"Generated newick: {end_time - start_time:.4f} seconds.")

prog_end_time = time.perf_counter()
print(f"Program run time: {prog_end_time - prog_time:.4f} seconds.")
# plt.figure()
# dendrogram = sp.cluster.hierarchy.dendrogram(linkage)
# plt.savefig("out.png")
