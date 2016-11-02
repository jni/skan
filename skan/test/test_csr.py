import os, sys
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from skan import csr

rundir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(rundir)

from skan._testdata import tinycycle, skeleton1


def test_tiny_cycle():
    g, idxs, degimg = csr.skeleton_to_csgraph(tinycycle)
    expected_indptr = [0, 0, 2, 4, 6, 8]
    expected_indices = [2, 3, 1, 4, 1, 4, 2, 3]
    expected_data = np.sqrt(2)

    assert_equal(g.indptr, expected_indptr)
    assert_equal(g.indices, expected_indices)
    assert_almost_equal(g.data, expected_data)

    expected_degrees = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]])
    assert_equal(degimg, expected_degrees)
    assert_equal(idxs, [0, 1, 3, 5, 7])


def test_skeleton1_stats():
    args = csr.skeleton_to_csgraph(skeleton1)
    stats = csr.branch_statistics(*args)

