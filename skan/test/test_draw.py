"""
Basic testing of the draw module. This just ensures the functions don't crash.
Testing plotting is hard. ;)
"""
import os
from skimage import io, morphology
import pytest

from skan import pre, draw

rundir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(rundir, 'data')


@pytest.fixture
def test_image():
    image = io.imread(os.path.join(datadir, 'skeleton.tif'))
    return image


@pytest.fixture
def test_skeleton(test_image):
    thresholded = pre.threshold(test_image, sigma=2, radius=31, offset=-7)
    skeleton = morphology.skeletonize(thresholded)
    return skeleton


def test_overlay_skeleton(test_image, test_skeleton):
    draw.overlay_skeleton_2d(test_image, test_skeleton)
    draw.overlay_skeleton_2d(test_image, test_skeleton,
                                     image_cmap='viridis')


def test_overlay_euclidean_skeleton(test_image, test_skeleton):
    draw.overlay_euclidean_skeleton_2d(test_image, test_skeleton)
    draw.overlay_euclidean_skeleton_2d(test_image, test_skeleton,
                                       skeleton_color_source='branch-distance')


def test_pipeline_plot(test_image):
    draw.pipeline_plot(test_image, sigma=2, radius=31, offset=-7)