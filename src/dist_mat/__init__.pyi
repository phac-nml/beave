from typing import Annotated

import numpy
from numpy.typing import NDArray

def calc_dists(
    array: Annotated[NDArray[numpy.uint32], dict(shape=(None, None), order="C", device="cpu")],
    threads: int,
    scaled: bool,
    count_missing: bool,
) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu")]:
    """Calculate all pairwise distances between all profiles."""

def fast_match(
    array: Annotated[NDArray[numpy.uint32], dict(shape=(None, None), order="C", device="cpu")],
    threads: int,
    scaled: bool,
    count_missing: bool,
    query_length: int,
    threshold: float,
) -> Annotated[
    NDArray[numpy.void],
    dict(shape=(3, None), order="C", device="cpu"),
]:
    """Calculate pairwise distances of query profiles against all other profiles.

    Function returns an array of arrays, with 3 columns, column 1 and 2 contain a numpy.uint32.
    The values in columns 1 and 2 correspond to the index of the associated samples input names.
    The input names are typically stored in the dataframe containing the input profiles.
    Column 3 contains a numpy.float32 value which is the distance between the two corresponding
    samples.
    """

__all__ = ["calc_dists", "fast_match"]
