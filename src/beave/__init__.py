"""Declaration of module and C++ functions."""

from .beave_ext import (  # type: ignore[import-not-found, no-redef]
    __doc__,
    calc_dists,
    fast_match,
)

__all__ = ("__doc__", "calc_dists", "fast_match")
