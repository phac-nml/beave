"""Declaration of module and C++ functions."""

from .dist_mat_ext import (  # type: ignore[import-not-found, no-redef]
    __doc__,
    calc_dists,
    fast_match,
)

__all__ = ("__doc__", "calc_dists", "fast_match")
