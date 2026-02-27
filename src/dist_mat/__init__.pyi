from typing import Annotated

import numpy
from numpy.typing import NDArray

def calc_dists(
    array: Annotated[
        NDArray[numpy.uint32], dict(shape=(None, None), order="C", device="cpu")
    ],
    threads: int,
    scaled: bool,
    count_missing: bool,
) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu")]:
    """
    Calculated all pairwise distances between all profiles.
    """

__all__ = ["calc_dists"]
