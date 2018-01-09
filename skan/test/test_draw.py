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

from skan import pre, draw, csr, _testdata

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
    draw.overlay_skeleton_2d(test_image, test_skeleton,
                                     image_cmap='viridis')


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


def test_networkx_plot():
    g0, c0, _ = csr.skeleton_to_csgraph(_testdata.skeleton0)
    g1, c1, _ = csr.skeleton_to_csgraph(_testdata.skeleton1)
    fig, axes = plt.subplots(1, 2)
    draw.overlay_skeleton_networkx(g0, c0, image=_testdata.skeleton0,
                                   axis=axes[0])
    draw.overlay_skeleton_networkx(g1, c1, image=_testdata.skeleton1,
                                   axis=axes[1])
