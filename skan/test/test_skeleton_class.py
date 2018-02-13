import os, sys
from collections import defaultdict
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from skan.csr import Skeleton

from skan._testdata import (tinycycle, tinyline, skeleton0, skeleton1,
                            skeleton2, skeleton3d, topograph1d, skeleton4)


def test_skeleton1_stats():
    skeleton = Skeleton(skeleton1)
    assert skeleton.paths.shape == (4, np.sum(skeleton1) + 1)
    paths_list = skeleton.paths_list()
    reference_paths = [
        [8, 6, 1, 2, 3, 4, 5, 7, 11, 10, 13],
        [8, 9, 13],
        [8, 12, 14],
        [13, 15, 16, 17]
    ]
    d0 = 1 + np.sqrt(2)
    reference_distances = [5 * d0, d0, d0, 1 + d0]
    for path in reference_paths:
        assert path in paths_list or path[::-1] in paths_list
    assert_allclose(sorted(skeleton.path_lengths()),
                    sorted(reference_distances))


def test_tip_junction_edges():
    skeleton = Skeleton(skeleton4)
    reference_paths = [[1, 2], [2, 4, 5], [2, 7]]
    paths_list = skeleton.paths_list()
    for path in reference_paths:
        assert path in paths_list or path[::-1] in paths_list
