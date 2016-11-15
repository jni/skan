import os, sys
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from skan import csr

rundir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(rundir)

from skan._testdata import (tinycycle, tinyline, skeleton1, skeleton2,
                            skeleton3d, topograph1d)


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
    g, idxs, degimg = csr.skeleton_to_csgraph(skeleton1)
    stats = csr.branch_statistics(g, idxs, degimg)
    assert_equal(stats.shape, (4, 4))
    keys = map(tuple, stats[:, :2].astype(int))
    dists = stats[:, 2]
    types = stats[:, 3].astype(int)
    ids2dist = dict(zip(keys, dists))
    assert (13, 8) in ids2dist
    assert (8, 13) in ids2dist
    d0, d1 = sorted((ids2dist[(13, 8)], ids2dist[(8, 13)]))
    assert_almost_equal(d0, 1 + np.sqrt(2))
    assert_almost_equal(d1, 5*d0)
    assert_equal(np.bincount(types), [0, 2, 2])
    assert_almost_equal(np.unique(dists), [d0, 2 + np.sqrt(2), d1])


def test_3skeletons():
    df = csr.summarise(skeleton2)
    assert_almost_equal(np.unique(df['euclidean-distance']),
                        np.sqrt([5, 10]))
    assert_equal(np.unique(df['skeleton-id']), [1, 2])
    assert_equal(np.bincount(df['branch-type']), [0, 4, 4])


def test_summarise_spacing():
    df = csr.summarise(skeleton2)
    df2 = csr.summarise(skeleton2, spacing=2)
    assert_equal(np.array(df['node-id-0']), np.array(df2['node-id-0']))
    assert_almost_equal(np.array(df2['euclidean-distance']),
                        np.array(2 * df['euclidean-distance']))
    assert_almost_equal(np.array(df2['branch-distance']),
                        np.array(2 * df['branch-distance']))


def test_line():
    g, idxs, degimg = csr.skeleton_to_csgraph(tinyline)
    assert_equal(idxs, [0, 1, 2, 3])
    assert_equal(degimg, [0, 1, 2, 1, 0])
    assert_equal(g.shape, (4, 4))
    assert_equal(csr.branch_statistics(g, idxs, degimg), [[1, 3, 2, 0]])


def test_cycle_stats():
    stats = csr.branch_statistics(*csr.skeleton_to_csgraph(tinycycle),
                                  buffer_size_offset=1)
    assert_almost_equal(stats, [[1, 1, 4*np.sqrt(2), 3]])


def test_3d_spacing():
    g, idxs, degimg = csr.skeleton_to_csgraph(skeleton3d, spacing=[5, 1, 1])
    stats = csr.branch_statistics(g, idxs, degimg)
    assert_equal(stats.shape, (5, 4))
    assert_almost_equal(stats[0], [1, 11, 2 * np.sqrt(27), 1])
    assert_equal(np.unique(stats[:, 3].astype(int)), [1, 2, 3])


def test_topograph():
    g, idxs, degimg = csr.skeleton_to_csgraph(topograph1d)
    stats = csr.branch_statistics(g, idxs, degimg)
    assert stats.shape == (1, 4)
    assert_almost_equal(stats[0], [1, 3, 2 * np.sqrt(2), 0])


def test_topograph_summary():
    stats = csr.summarise(topograph1d, spacing=2.5)
    assert stats.loc[0, 'euclidean-distance'] == 5.0
    assert_almost_equal(stats.loc[0, ['coord-0-0', 'coord-0-1',
                                      'coord-1-0', 'coord-1-1']],
                        [3, 0, 3, 5])
