import os, sys
from collections import defaultdict
from time import process_time
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from skan.csr import Skeleton

from skan._testdata import (tinycycle, tinyline, skeleton0, skeleton1,
                            skeleton2, skeleton3d, topograph1d, skeleton4)


def test_skeleton1_topo():
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


def test_skeleton1_float():
    image = np.zeros(skeleton1.shape, dtype=float)
    image[skeleton1] = 1 + np.random.random(np.sum(skeleton1))
    skeleton = Skeleton(image)
    path, data = skeleton.path_with_data(0)
    assert 1.0 < np.mean(data) < 2.0


def test_path_length_caching():
    skeleton = Skeleton(skeleton3d)
    t0 = process_time()
    distances = skeleton.path_lengths()
    t1 = process_time()
    distances2 = skeleton.path_lengths()
    t2 = process_time()
    assert t2 - t1 < t1 - t0
    assert np.all((distances > 0.99 + np.sqrt(2))
                  & (distances < 5.01 + 5 * np.sqrt(2)))


def test_tip_junction_edges():
    skeleton = Skeleton(skeleton4)
    reference_paths = [[1, 2], [2, 4, 5], [2, 7]]
    paths_list = skeleton.paths_list()
    for path in reference_paths:
        assert path in paths_list or path[::-1] in paths_list
