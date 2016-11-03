import os, sys
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from skan import nx

rundir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(rundir)

from skan._testdata import tinycycle, skeleton1, skeleton2


def test_tiny_cycle():
    g, degimg, skel_labels = nx.skeleton_to_nx(tinycycle)
    expected_edges = [(1, 2), (1, 3), (2, 4), (3, 4)]

    assert sorted(g.edges()) == expected_edges

    assert_almost_equal([g[a][b]['weight'] for a, b in g.edges()], np.sqrt(2))

    expected_degrees = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]])
    assert_equal(degimg, expected_degrees)

    assert all(g.node[n]['type'] == 'path' for n in g)


def test_skeleton1_stats():
    g = nx.skeleton_to_nx(skeleton1)[0]
    stats = nx.branch_statistics(g)
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
    df = nx.summarise(skeleton2)
    assert_almost_equal(np.unique(df['euclidean-distance']),
                        np.sqrt([5, 10]))
    assert_equal(np.unique(df['skeleton-id']), [0, 1])
    assert_equal(np.bincount(df['branch-type']), [0, 4, 4])
