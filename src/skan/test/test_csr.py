from collections import defaultdict

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from skimage.draw import line

from skan import csr
from skan._testdata import (
        tinycycle, tinyline, skeleton0, skeleton1, skeleton2, skeleton3d,
        topograph1d, skeleton4
        )


def _old_branch_statistics(
        skeleton_image, *, spacing=1, value_is_height=False
        ):
    skel = csr.Skeleton(
            skeleton_image, spacing=spacing, value_is_height=value_is_height
            )
    summary = csr.summarize(skel, value_is_height=value_is_height)
    columns = ['node-id-src', 'node-id-dst', 'branch-distance', 'branch-type']
    return summary[columns].to_numpy()


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
    stats = _old_branch_statistics(skeleton1)
    assert_equal(stats.shape, (4, 4))
    keys = map(tuple, np.sort(stats[:, :2].astype(int), axis=1))
    dists = stats[:, 2]
    types = stats[:, 3].astype(int)
    ids2dist = defaultdict(list)
    for key, dist in zip(keys, dists):
        ids2dist[key].append(dist)
    assert (7, 12) in ids2dist
    d0, d1 = sorted(ids2dist[(7, 12)])
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
    assert_equal(_old_branch_statistics(tinyline), [[0, 2, 2, 0]])


def test_cycle_stats():
    stats = _old_branch_statistics(tinycycle)
    # source, dest, length, type
    assert_almost_equal(stats, [[0, 0, 4 * np.sqrt(2), 3]])


def test_3d_spacing():
    stats = _old_branch_statistics(skeleton3d, spacing=[5, 1, 1])
    assert_equal(stats.shape, (7, 4))
    assert_almost_equal(stats[0], [0, 10, 2 * np.sqrt(5**2 + 1 + 1), 1])
    # source, dest, length, type
    # test only junction-tip segments
    assert_equal(np.unique(stats[:, 3].astype(int)), [1, 2, 3])


def test_topograph():
    stats = _old_branch_statistics(topograph1d, value_is_height=True)
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
    g, _ = csr.skeleton_to_csgraph(skeleton0)
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


def test_transpose_image():
    image = np.zeros((10, 10))

    rr, cc = line(4, 0, 4, 2)
    image[rr, cc] = 1
    rr, cc = line(3, 2, 3, 5)
    image[rr, cc] = 1
    rr, cc = line(1, 2, 8, 2)
    image[rr, cc] = 1
    rr, cc = line(1, 0, 1, 8)
    image[rr, cc] = 1

    skeleton1 = csr.Skeleton(image)
    skeleton2 = csr.Skeleton(image.T)

    assert (skeleton1.n_paths == skeleton2.n_paths)
    np.testing.assert_allclose(
            np.sort(skeleton1.path_lengths()),
            np.sort(skeleton2.path_lengths()),
            )


def test_fast_graph_center_idx():
    s = csr.Skeleton(skeleton0)
    i = csr._fast_graph_center_idx(s)
    assert i == 6

    s = csr.Skeleton(skeleton4)
    i = csr._fast_graph_center_idx(s)
    assert i == 1

def test_sholl():
    s = csr.Skeleton(skeleton0)
    c, r, counts = csr.sholl_analysis(s, shells=np.arange(0, 5, 1.5))
    np.testing.assert_equal(c, [3, 3])
    np.testing.assert_equal(counts, [0, 3, 3, 0])


def test_sholl_spacing():
    s = csr.Skeleton(skeleton0, spacing=(1, 5))
    with pytest.warns(UserWarning):
        c, r, counts = csr.sholl_analysis(
                s, center=[3, 15], shells=np.arange(17)
                )
        for i in range(4):
            assert np.isin(i, counts)
    c, r, counts = csr.sholl_analysis(
            s, center=[3, 15], shells=np.arange(1, 20, 6)
            )
    np.testing.assert_equal(counts, [3, 2, 2, 0])


def test_diagonal():
    s = csr.Skeleton(skeleton4)
    # We choose the shells so that we catch all three, then two, then one arm
    # of the skeleton, while not triggering the "shell spacing too small"
    # warning
    c, r, counts = csr.sholl_analysis(
            s, center=[1, 1], shells=np.arange(0.09, 5, 1.45)
            )
    np.testing.assert_equal(counts, [3, 2, 1, 0])


def test_zero_degree_nodes():
    """Test that graphs with 0-degree nodes don't have allocation errors.

    In skan commits 7498831 or prior, isolated pixels, which have degree 0,
    would count as *negative* values when counting the total number of edges,
    resulting in a buffer too small to hold the edge info.

    See issue jni/skan#182.
    """
    x = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]).astype(bool)

    # the try/except causes an early segfault if the buffer overflows
    try:
        s = csr.Skeleton(x)
    except Exception as e:
        print(e)

    assert s.n_paths == 2
    np.testing.assert_equal(
            np.ravel(s.coordinates), [0, 1, 3, 5, 7, 9, 11, 12]
            )
