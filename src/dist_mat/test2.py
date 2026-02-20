#!/usr/bin/env python3
import numpy as np
import scipy as sp
import dist_mat as dm


array = np.array(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 9]],
    dtype=np.uint32,
)

sp_mat = sp.spatial.distance.pdist(array, metric="hamming")
print(sp_mat)
print(dm.calc_dists(array, 1, True, False))
