import os, sys
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from skan import csr

rundir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(rundir)

from skan._testdata import (tinycycle, tinyline, skeleton0, skeleton1,
                            skeleton2, skeleton3d, topograph1d, skeleton4)


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
    assert_equal(np.ravel_multi_index(idxs.astype(int).T, tinycycle.shape),
                 [0, 1, 3, 5, 7])


def test_skeleton1_stats():
    g, idxs, degimg = csr.skeleton_to_csgraph(skeleton1)
    stats = csr.branch_statistics(g)
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
    assert_equal(np.ravel(idxs), [0, 1, 2, 3])
    assert_equal(degimg, [0, 1, 2, 1, 0])
    assert_equal(g.shape, (4, 4))
    assert_equal(csr.branch_statistics(g), [[1, 3, 2, 0]])


def test_cycle_stats():
    stats = csr.branch_statistics(csr.skeleton_to_csgraph(tinycycle)[0],
                                  buffer_size_offset=1)
    assert_almost_equal(stats, [[1, 1, 4*np.sqrt(2), 3]])


def test_3d_spacing():
    g, idxs, degimg = csr.skeleton_to_csgraph(skeleton3d, spacing=[5, 1, 1])
    stats = csr.branch_statistics(g)
    assert_equal(stats.shape, (5, 4))
    assert_almost_equal(stats[0], [1, 5, 10.467, 1], decimal=3)
    assert_equal(np.unique(stats[:, 3].astype(int)), [1, 2, 3])


def test_topograph():
    g, idxs, degimg = csr.skeleton_to_csgraph(topograph1d,
                                              value_is_height=True)
    stats = csr.branch_statistics(g)
    assert stats.shape == (1, 4)
    assert_almost_equal(stats[0], [1, 3, 2 * np.sqrt(2), 0])


def test_topograph_summary():
    stats = csr.summarise(topograph1d, spacing=2.5, using_height=True)
    assert stats.loc[0, 'euclidean-distance'] == 5.0
    assert_almost_equal(stats.loc[0, ['coord-src-0', 'coord-src-1',
                                      'coord-dst-0', 'coord-dst-1']],
                        [3, 0, 3, 5])


def test_junction_multiplicity():
    """Test correct distances when a junction has more than one pixel."""
    g, idxs, degimg = csr.skeleton_to_csgraph(skeleton0)
    assert_almost_equal(g[3, 5], 2.0155644)
    g, idxs, degimg = csr.skeleton_to_csgraph(skeleton0,
                                              unique_junctions=False)
    assert_almost_equal(g[2, 3], 1.0)
    assert_almost_equal(g[3, 6], np.sqrt(2))


def test_multiplicity_stats():
    stats1 = csr.summarise(skeleton0)
    stats2 = csr.summarise(skeleton0, spacing=2)
    assert_almost_equal(2 * stats1['branch-distance'].values,
                        stats2['branch-distance'].values)
    assert_almost_equal(2 * stats1['euclidean-distance'].values,
                        stats2['euclidean-distance'].values)


def test_pixel_values():
    image = np.random.random((45,))
    expected = np.mean(image[1:-1])
    stats = csr.summarise(image)
    assert_almost_equal(stats.loc[0, 'mean pixel value'], expected)


def test_tip_junction_edges():
    stats1 = csr.summarise(skeleton4)
    assert stats1.shape[0] == 3  # ensure all three branches are counted
