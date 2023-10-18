from collections import defaultdict

import pytest
import networkx as nx
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_equal, assert_almost_equal
import pandas as pd
from skimage.draw import line

from skan import csr, summarize
from skan._testdata import (
        tinycycle,
        tinyline,
        skeleton0,
        skeleton1,
        skeleton2,
        skeleton3d,
        topograph1d,
        skeleton4,
        skeleton_loop1,
        skeleton_loop2,
        skeleton_linear1,
        skeleton_linear2,
        skeleton_linear3,
        )


def _old_branch_statistics(
        skeleton_image, *, spacing=1, value_is_height=False
        ):
    skel = csr.Skeleton(
            skeleton_image, spacing=spacing, value_is_height=value_is_height
            )
    summary = csr.summarize(skel, value_is_height=value_is_height)
    columns = ["node-id-src", "node-id-dst", "branch-distance", "branch-type"]
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
    assert_almost_equal(np.unique(df["euclidean-distance"]), np.sqrt([5, 10]))
    assert_equal(np.unique(df["skeleton-id"]), [0, 1])
    assert_equal(np.bincount(df["branch-type"]), [0, 4, 4])


def test_summarize_spacing():
    df = csr.summarize(csr.Skeleton(skeleton2))
    df2 = csr.summarize(csr.Skeleton(skeleton2, spacing=2))
    assert_equal(np.array(df["node-id-src"]), np.array(df2["node-id-src"]))
    assert_almost_equal(
            np.array(df2["euclidean-distance"]),
            np.array(2 * df["euclidean-distance"])
            )
    assert_almost_equal(
            np.array(df2["branch-distance"]),
            np.array(2 * df["branch-distance"])
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
    assert stats.loc[0, "euclidean-distance"] == 5.0
    columns = ["coord-src-0", "coord-src-1", "coord-dst-0", "coord-dst-1"]
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
            2 * stats1["branch-distance"].values,
            stats2["branch-distance"].values
            )
    assert_almost_equal(
            2 * stats1["euclidean-distance"].values,
            stats2["euclidean-distance"].values
            )


def test_pixel_values():
    image = np.random.random((45,))
    expected = np.mean(image)
    stats = csr.summarize(csr.Skeleton(image))
    assert_almost_equal(stats.loc[0, "mean-pixel-value"], expected)


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

    assert skeleton1.n_paths == skeleton2.n_paths
    np.testing.assert_allclose(
            np.sort(skeleton1.path_lengths()),
            np.sort(skeleton2.path_lengths()),
            )


@pytest.mark.parametrize(
        "skeleton,prune_branch,target",
        [
                (
                        skeleton1,
                        1,
                        np.array([
                                [0, 1, 1, 1, 1, 1, 0],
                                [1, 0, 0, 0, 0, 0, 1],
                                [0, 1, 1, 0, 1, 1, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                ]),
                        ),
                (
                        skeleton1,
                        2,
                        np.array([
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0],
                                [1, 0, 0, 2, 0, 0, 0],
                                [1, 0, 0, 0, 2, 2, 2],
                                ]),
                        ),
                # There are no isolated cycles to be pruned
                (
                        skeleton1,
                        3,
                        np.array([
                                [0, 1, 1, 1, 1, 1, 0],
                                [1, 0, 0, 0, 0, 0, 1],
                                [0, 3, 2, 0, 1, 1, 0],
                                [3, 0, 0, 4, 0, 0, 0],
                                [3, 0, 0, 0, 4, 4, 4],
                                ]),
                        ),
                ],
        )
def test_prune_paths(
        skeleton: np.ndarray, prune_branch: int, target: np.ndarray
        ) -> None:
    """Test pruning of paths."""
    s = csr.Skeleton(skeleton, keep_images=True)
    summary = summarize(s)
    indices_to_remove = summary.loc[summary["branch-type"] == prune_branch
                                    ].index
    pruned = s.prune_paths(indices_to_remove)
    np.testing.assert_array_equal(pruned, target)


def test_prune_paths_exception_single_point() -> None:
    """Test exceptions raised when pruning leaves a single point and Skeleton object
    can not be created and returned."""
    s = csr.Skeleton(skeleton0)
    summary = summarize(s)
    indices_to_remove = summary.loc[summary["branch-type"] == 1].index
    with pytest.raises(ValueError):
        s.prune_paths(indices_to_remove)


def test_prune_paths_exception_invalid_path_index() -> None:
    """Test exceptions raised when trying to prune paths that do not exist in the summary. This can arise if skeletons
    are not updated correctly during iterative pruning."""
    s = csr.Skeleton(skeleton0)
    summary = summarize(s)
    indices_to_remove = [6]
    with pytest.raises(ValueError):
        s.prune_paths(indices_to_remove)


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


@pytest.mark.parametrize(
        "skeleton, paths, branch_type, branch_distance, euclidean_distance",
        [
                [skeleton_loop1, 1, 3, 159.23759005323606, 0],
                [skeleton_loop2, 1, 3, 161.33809511662434, 0],
                [
                        skeleton_linear1, 1, 0, 154.09545442950505,
                        114.84337159801605
                        ],
                [skeleton_linear2, 1, 0, 84.15432893255064, 71.30918594402827],
                ],
        )
def test_iteratively_prune_paths(
        skeleton: np.ndarray, paths: int, branch_type: int,
        branch_distance: float, euclidean_distance: float
        ) -> None:
    """Test iteratively pruning a skeleton."""
    pruned_skeleton = csr.iteratively_prune_paths(skeleton)
    skeleton_summary = csr.summarize(pruned_skeleton)
    assert isinstance(pruned_skeleton, csr.Skeleton)
    assert skeleton_summary.shape[0] == paths
    assert skeleton_summary["branch-type"][0] == branch_type
    assert skeleton_summary["branch-distance"][0] == branch_distance
    assert skeleton_summary["euclidean-distance"][0] == euclidean_distance


@pytest.mark.parametrize(
        "skeleton, paths, branch_type, branch_distance, euclidean_distance",
        [[
                skeleton_linear3,
                3,
                [0, 0, 0],
                [164.05382386916244, 20.656854249492383, 29.485281374238575],
                [110.11357772772621, 19.4164878389476, 24.186773244895647],
                ]],
        )
def test_iteratively_prune_multiple_paths(
        skeleton: np.ndarray, paths: int, branch_type: int,
        branch_distance: float, euclidean_distance: float
        ) -> None:
    """Test iteratively pruning a image with multiple skeletons."""
    pruned_skeleton = csr.iteratively_prune_paths(skeleton)
    skeleton_summary = csr.summarize(pruned_skeleton)
    assert isinstance(pruned_skeleton, csr.Skeleton)
    assert skeleton_summary.shape[0] == paths
    assert list(skeleton_summary["branch-type"]) == branch_type
    assert list(skeleton_summary["branch-distance"]) == branch_distance
    assert list(skeleton_summary["euclidean-distance"]) == euclidean_distance


def test_skeleton_path_image_no_keep_image():
    """See #208: "Skeleton.path_label_image requires keep_images=True."

    Before PR #210, it was an implicit requirement of path_label_image
    to create a Skeleton with keep_images=True. However, we only needed
    the shape of the image. This makes sure that the method works even
    when keep_images=False.
    """
    s = csr.Skeleton(skeleton2, keep_images=False)
    pli = s.path_label_image()
    assert np.max(pli) == s.n_paths


@pytest.mark.parametrize(
        ("np_skeleton", "n"),
        [
                (tinycycle, 3),
                #        (tinyline, 5),
                (skeleton0, 7),
                (skeleton1, 7),
                (skeleton2, 14),
                (np.random.choice((0, 1), size=(2, 18)), 18),
                (np.random.choice((0, 1), size=(37, 4)), 37),
                (np.random.choice((0, 1), size=(42, 42)), 42),
                ]
        )
def test_pad_non_square_arrays(np_skeleton: npt.NDArray, n: int) -> None:
    """Test padding of non-square arrays."""
    padded_array = csr._pad_non_square_arrays(np_skeleton)
    padded_nrows, padded_ncols = padded_array.shape
    assert padded_nrows == n
    assert padded_ncols == n


@pytest.mark.parametrize(
        ("np_skeleton", "summary", "nodes", "edges"),
        [
                (skeleton0, summarize(csr.Skeleton(skeleton0)), 4, 3),
                # (skeleton1, None, 4, 3),
                # (skeleton1, summarize(csr.Skeleton(skeleton1)), 4, 3),
                # (skeleton2, summarize(csr.Skeleton(skeleton2)), 8, 6),
                # (skeleton3d, None, 7, 7),
                # (skeleton_loop1, None, 10, 10),
                # (skeleton_linear1, None, 24, 23),
                ],
        )
def test_skeleton_to_nx(
        np_skeleton: npt.NDArray, summary: pd.DataFrame, nodes: int, edges: int
        ) -> None:
    """Test creation of NetworkX Graph from skeletons arrays and summary."""
    print(f"np_skeleton : \n{np_skeleton}")
    skeleton = csr.Skeleton(np_skeleton)
    # skan_nx = csr.skeleton_to_nx(skeleton, summary)
    skan_nx = csr.skeleton_to_nx(skeleton.skeleton_image)
    assert skan_nx.number_of_nodes() == nodes
    assert skan_nx.number_of_edges() == edges
    assert False


@pytest.mark.parametrize("np_skeleton", [(skeleton0)])
def test_nx_to_skeleton(np_skeleton: npt.NDArray) -> None:
    """Test converting Networkx graph to Skeleton object."""
    skeleton = csr.Skeleton(np_skeleton)
    summary = summarize(skeleton)
    # skan_nx = csr.skeleton_to_nx(skeleton, summary)
    skan_nx = csr.skeleton_to_nx(skeleton.skeleton_image)
    new_skeleton = csr.nx_to_skeleton(skan_nx)
    assert np.testing.assert_array_equal(
            new_skeleton, csr._pad_non_square_arrays(np_skeleton)
            )
