import sys
from time import process_time
import numpy as np
from numpy.testing import assert_allclose
import pytest
from skan.csr import Skeleton, summarize

from skan._testdata import (
        tinycycle, skeleton1, skeleton2, skeleton3d, skeleton4, junction_first
        )


def test_tiny_cycle():
    skeleton = Skeleton(tinycycle)
    assert skeleton.paths.shape == (1, 5)


def test_skeleton1_topo():
    skeleton = Skeleton(skeleton1)
    assert skeleton.paths.shape == (4, 21)
    paths_list = skeleton.paths_list()
    reference_paths = [[7, 5, 0, 1, 2, 3, 4, 6, 10, 9, 12], [7, 8, 12],
                       [7, 11, 13], [12, 14, 15, 16]]
    d0 = 1 + np.sqrt(2)
    reference_distances = [5 * d0, d0, d0, 1 + d0]
    for path in reference_paths:
        assert path in paths_list or path[::-1] in paths_list
    assert_allclose(
            sorted(skeleton.path_lengths()), sorted(reference_distances)
            )


def test_skeleton1_float():
    image = np.zeros(skeleton1.shape, dtype=float)
    image[skeleton1] = 1 + np.random.random(np.sum(skeleton1))
    skeleton = Skeleton(image)
    path, data = skeleton.path_with_data(0)
    assert 1.0 < np.mean(data) < 2.0


def test_skeleton_coordinates():
    skeleton = Skeleton(skeleton1)
    last_path_coordinates = skeleton.path_coordinates(3)
    assert_allclose(last_path_coordinates, [[3, 3], [4, 4], [4, 5], [4, 6]])


@pytest.mark.skipif(
        sys.platform.startswith('win'),
        reason='caching not working on Windows CI for mysterious reasons',
        )
def test_path_length_caching():
    skeleton = Skeleton(skeleton3d)
    t0 = process_time()
    distances = skeleton.path_lengths()
    t1 = process_time()
    distances2 = skeleton.path_lengths()
    t2 = process_time()
    assert t2 - t1 < t1 - t0
    assert np.all((distances >= np.sqrt(2))
                  & (distances < 5.01 + 5 * np.sqrt(2)))


def test_tip_junction_edges():
    skeleton = Skeleton(skeleton4)
    reference_paths = [[1, 0], [1, 2, 3, 4], [1, 5, 6]]
    paths_list = skeleton.paths_list()
    for path in reference_paths:
        assert path in paths_list or path[::-1] in paths_list


def test_path_stdev():
    image = np.zeros(skeleton1.shape, dtype=float)
    image[skeleton1] = 1 + np.random.random(np.sum(skeleton1))
    skeleton = Skeleton(image)
    # longest_path should be 0, but could change.
    longest_path = np.argmax(skeleton.path_lengths())
    dev = skeleton.path_stdev()[longest_path]
    assert 0.09 < dev < 0.44  # chance is < 1/10K that this will fail

    # second check: first principles.
    skeleton2 = Skeleton(image**2)
    # (Var = StDev**2 = E(X**2) - (E(X))**2)
    assert_allclose(
            skeleton.path_stdev()**2,
            skeleton2.path_means() - skeleton.path_means()**2
            )


def test_junction_first():
    """Ensure no self-edges exist in multi-pixel junctions.

    Before commit 64047622, the skeleton class would include self-edges
    within junctions in its paths list, but only when the junction was visited
    before any of its adjacent branches. This turns out to be tricky to achieve
    but not impossible in 2D.
    """
    assert [1, 1] not in Skeleton(junction_first).paths_list()


def test_skeleton_summarize():
    image = np.zeros(skeleton2.shape, dtype=float)
    image[skeleton2] = 1 + np.random.random(np.sum(skeleton2))
    skeleton = Skeleton(image)
    summary = summarize(skeleton)
    assert set(summary['skeleton-id']) == {0, 1}
    assert (
            np.all(summary['mean-pixel-value'] < 2)
            and np.all(summary['mean-pixel-value'] > 1)
            )


def test_skeleton_label_image_strict():
    """Test that the skeleton pixel labels match the branch IDs.
    
    This does pixel-wise pairing of the label image with the expected label
    image.  There should be the same number of unique pairs as there are unique
    labels in the expected label image. This assumes that the branches are
    displayed correctly, but does not assert an order to the numbering of the
    branches. Junction pixel IDs are ambiguous and so are not checked. It is an
    implementation detail that currently, the latest branch ID touching that
    junction "wins".
    """
    skeleton = Skeleton(skeleton4)
    label_image = np.asarray(skeleton)
    expected = np.array([
            [1, 0, 0, 0, 0],
            [0, 3, 2, 2, 2],
            [0, 3, 0, 0, 0],
            [0, 3, 0, 0, 0],
            ])
    non_junction_pixels = (skeleton.degrees < 3)
    non_junction_coords = tuple(
            np.transpose(skeleton.coordinates[non_junction_pixels])
            )
    paired_values = np.stack(
            (expected[non_junction_coords], label_image[non_junction_coords]),
            axis=-1
            ).reshape((-1, 2))
    unique_pairs = np.unique(paired_values, axis=0)
    # check that the number of pairs is equal to the number of values
    expected_label_values = np.unique(expected[expected > 0])
    assert len(expected_label_values) == len(unique_pairs)
    # check that all nonzero pixels are preserved
    np.testing.assert_array_equal(expected != 0, label_image != 0)


def test_skeleton_label_image():
    """Simple test that the skeleton label image covers the same
    pixels as the expected label image.
    """
    skeleton = Skeleton(skeleton4)
    label_image = np.asarray(skeleton)
    expected = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 2, 2, 2],
            [0, 3, 0, 0, 0],
            [0, 3, 0, 0, 0],
            ])

    np.testing.assert_array_equal(
            label_image.astype(bool), expected.astype(bool)
            )
