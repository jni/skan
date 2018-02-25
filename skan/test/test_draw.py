"""
Basic testing of the draw module. This just ensures the functions don't crash.
Testing plotting is hard. ;)
"""
import os
import numpy as np
from skimage import io, morphology
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

from skan import pre, draw, csr, _testdata, Skeleton

rundir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(rundir, 'data')


@pytest.fixture
def test_image():
    image = io.imread(os.path.join(datadir, 'skeleton.tif'))
    return image


@pytest.fixture
def test_thresholded(test_image):
    thresholded = pre.threshold(test_image, sigma=2, radius=31, offset=0.075)
    return thresholded


@pytest.fixture
def test_skeleton(test_thresholded):
    skeleton = morphology.skeletonize(test_thresholded)
    return skeleton


@pytest.fixture
def test_stats(test_skeleton):
    stats = csr.summarise(test_skeleton)
    return stats


def test_overlay_skeleton(test_image, test_skeleton):
    draw.overlay_skeleton_2d(test_image, test_skeleton)
    draw.overlay_skeleton_2d(test_image, test_skeleton, image_cmap='viridis')
    draw.overlay_skeleton_2d(test_image, test_skeleton, dilate=1)


def test_overlay_euclidean_skeleton(test_image, test_stats):
    draw.overlay_euclidean_skeleton_2d(test_image, test_stats)
    draw.overlay_euclidean_skeleton_2d(test_image, test_stats,
                                       skeleton_color_source='branch-distance')


def test_pipeline_plot(test_image, test_thresholded, test_skeleton,
                       test_stats):
    draw.pipeline_plot(test_image, test_thresholded, test_skeleton,
                       test_stats)


def test_pipeline_plot_existing_fig(test_image, test_thresholded,
                                    test_skeleton, test_stats):
    fig = Figure()
    draw.pipeline_plot(test_image, test_thresholded, test_skeleton, test_stats,
                       figure=fig)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    draw.pipeline_plot(test_image, test_thresholded, test_skeleton, test_stats,
                       figure=fig, axes=np.ravel(axes))


def test_skeleton_class_overlay(test_image, test_skeleton):
    fig, axes = plt.subplots()
    skeleton = Skeleton(test_skeleton, source_image=test_image)
    draw.overlay_skeleton_2d_class(skeleton,
                                   skeleton_color_source='path_lengths')
    def filtered(skeleton):
        means = skeleton.path_means()
        low = means < 0.125
        just_right = (0.125 < means) & (means < 0.625)
        high = 0.625 < means
        return 0 * low + 1 * just_right + 2 * high
    fig, ax = plt.subplots()
    draw.overlay_skeleton_2d_class(skeleton,
                                   skeleton_color_source=filtered,
                                   vmin=0, vmax=2, axes=ax)
    with pytest.raises(ValueError):
        draw.overlay_skeleton_2d_class(skeleton,
                                       skeleton_color_source='filtered')



def test_networkx_plot():
    g0, c0, _ = csr.skeleton_to_csgraph(_testdata.skeleton0)
    g1, c1, _ = csr.skeleton_to_csgraph(_testdata.skeleton1)
    fig, axes = plt.subplots(1, 2)
    draw.overlay_skeleton_networkx(g0, c0, image=_testdata.skeleton0,
                                   axis=axes[0])
    draw.overlay_skeleton_networkx(g1, c1, image=_testdata.skeleton1,
                                   axis=axes[1])
    # test axis=None and additional coordinates
    c2 = np.concatenate((c1, np.random.random(c1[:1].shape)), axis=0)
    draw.overlay_skeleton_networkx(g1, c2, image=_testdata.skeleton1)
