import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pytest
from skan import csr
from skan._testdata import (
        tinycycle, tinyline, skeleton0, skeleton1, skeleton2, skeleton3d,
        topograph1d, skeleton4
        )


def test_tiny_cycle():
    g, idxs = csr.skeleton_to_csgraph(tinycycle)
    expected_indptr = [0, 2, 4, 6, 8]
    expected_indices = [1, 2, 0, 3, 0, 3, 1, 2]
    expected_data = np.sqrt(2)

    assert_equal(g.indptr, expected_indptr)
    assert_equal(g.indices, expected_indices)
    assert_almost_equal(g.data, expected_data)

    assert_equal(np.ravel_multi_index(idxs, tinycycle.shape), [1, 3, 5, 7])


def test_skeleton1_stats():
    g, idxs = csr.skeleton_to_csgraph(skeleton1)
    stats = csr.branch_statistics(g)
    assert_equal(stats.shape, (4, 4))
    keys = map(tuple, stats[:, :2].astype(int))
    dists = stats[:, 2]
    types = stats[:, 3].astype(int)
    ids2dist = dict(zip(keys, dists))
    assert (12, 7) in ids2dist
    assert (7, 12) in ids2dist
    d0, d1 = sorted((ids2dist[(12, 7)], ids2dist[(7, 12)]))
    assert_almost_equal(d0, 1 + np.sqrt(2))
    assert_almost_equal(d1, 5 * d0)
    assert_equal(np.bincount(types), [0, 2, 2])
    assert_almost_equal(np.unique(dists), [d0, 2 + np.sqrt(2), d1])


def test_2skeletons():
    df = csr.summarize(csr.Skeleton(skeleton2))
    assert_almost_equal(np.unique(df['euclidean-distance']), np.sqrt([5, 10]))
    assert_equal(np.unique(df['skeleton-id']), [0, 1])
    assert_equal(np.bincount(df['branch-type']), [0, 4, 4])


def test_summarize_spacing():
    df = csr.summarize(csr.Skeleton(skeleton2))
    df2 = csr.summarize(csr.Skeleton(skeleton2, spacing=2))
    assert_equal(np.array(df['node-id-src']), np.array(df2['node-id-src']))
    assert_almost_equal(
            np.array(df2['euclidean-distance']),
            np.array(2 * df['euclidean-distance'])
            )
    assert_almost_equal(
            np.array(df2['branch-distance']),
            np.array(2 * df['branch-distance'])
            )


def test_line():
    g, idxs = csr.skeleton_to_csgraph(tinyline)
    assert_equal(np.ravel(idxs), [1, 2, 3])
    assert_equal(g.shape, (3, 3))
    # source, dest, length, type
    assert_equal(csr.branch_statistics(g), [[0, 2, 2, 0]])


def test_cycle_stats():
    stats = csr.branch_statistics(
            csr.skeleton_to_csgraph(tinycycle)[0], buffer_size_offset=1
            )
    # source, dest, length, type
    assert_almost_equal(stats, [[0, 0, 4 * np.sqrt(2), 3]])


def test_3d_spacing():
    g, idxs = csr.skeleton_to_csgraph(skeleton3d, spacing=[5, 1, 1])
    stats = csr.branch_statistics(g)
    assert_equal(stats.shape, (5, 4))
    assert_almost_equal(stats[0], [0, 10, 2 * np.sqrt(5**2 + 1 + 1), 1])
    # source, dest, length, type
    # test only junction-tip segments
    assert_equal(np.unique(stats[:, 3].astype(int)), [1, 2, 3])


def test_topograph():
    g, idxs = csr.skeleton_to_csgraph(topograph1d, value_is_height=True)
    stats = csr.branch_statistics(g)
    assert stats.shape == (1, 4)
    assert_almost_equal(stats[0], [0, 2, 2 * np.sqrt(2), 0])


def test_topograph_summary():
    stats = csr.summarize(
            csr.Skeleton(topograph1d, spacing=2.5, value_is_height=True),
            value_is_height=True,
            )
    assert stats.loc[0, 'euclidean-distance'] == 5.0
    columns = ['coord-src-0', 'coord-src-1', 'coord-dst-0', 'coord-dst-1']
    assert_almost_equal(sorted(stats.loc[0, columns]), [0, 3, 3, 5])


def test_junction_multiplicity():
    """Test correct distances when a junction has more than one pixel."""
    g, idxs = csr.skeleton_to_csgraph(skeleton0)
    assert_equal(g.data, 1.0)
    assert_equal(g[2, 5], 0.0)


def test_multiplicity_stats():
    stats1 = csr.summarize(csr.Skeleton(skeleton0))
    stats2 = csr.summarize(csr.Skeleton(skeleton0, spacing=2))
    assert_almost_equal(
            2 * stats1['branch-distance'].values,
            stats2['branch-distance'].values
            )
    assert_almost_equal(
            2 * stats1['euclidean-distance'].values,
            stats2['euclidean-distance'].values
            )


def test_pixel_values():
    image = np.random.random((45,))
    expected = np.mean(image)
    stats = csr.summarize(csr.Skeleton(image))
    assert_almost_equal(stats.loc[0, 'mean-pixel-value'], expected)


def test_tip_junction_edges():
    stats1 = csr.summarize(csr.Skeleton(skeleton4))
    assert stats1.shape[0] == 3  # ensure all three branches are counted


def test_mst_junctions():
    g, _ = csr.skeleton_to_csgraph(skeleton0)
    h = csr._mst_junctions(g)
    hprime, _ = csr.skeleton_to_csgraph(skeleton0)

    G = g.todense()
    G[G > 1.1] = 0

    np.testing.assert_equal(G, h.todense())
    np.testing.assert_equal(G, hprime.todense())
