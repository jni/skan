import os
import imageio
from scipy import special
import numpy as np
from hypothesis import given, strategies
import pytest

from skan import pre


@pytest.fixture
def image():
    rundir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(rundir, 'data')
    image = imageio.imread(os.path.join(datadir, 'retic.tif'), format='fei')
    return image


def ball_volume(ndim, radius):
    """Return the volume of a ball of dimension `ndim` and radius `radius`."""
    n = ndim
    r = radius
    return np.pi ** (n / 2) / special.gamma(n / 2 + 1) * r ** n


@given(strategies.integers(min_value=1, max_value=4),
       strategies.integers(min_value=2, max_value=10))
def test_hyperball_volume(ndim, radius):
    theoretical_volume = ball_volume(ndim, radius)
    approximate_volume = np.sum(pre.hyperball(ndim, radius))
    np.testing.assert_allclose(approximate_volume, theoretical_volume,
                               rtol=0.5)


def test_threshold2d_sauvola(image):
    res = image.meta['Scan']['PixelHeight'] * 1e9  # nm/pixel
    radius = int(50 / res)  # radius of 50nm in pixels
    sigma = 0.1 * radius
    thresholded0 = pre.threshold(image, sigma=sigma, radius=radius,
                                 method='sauvola', offset=0.2)
    assert thresholded0.dtype == bool
    assert thresholded0.shape == image.shape
    thresholded1 = pre.threshold(image, sigma=sigma, radius=radius,
                                 method='sauvola', offset=0.075)
    assert np.all(thresholded1 <= thresholded0)


def test_threshold_2d_niblack(image):
    res = image.meta['Scan']['PixelHeight'] * 1e9  # nm/pixel
    radius = int(50 / res)
    sigma = 0.1 * radius
    thresholded0 = pre.threshold(image, sigma=sigma, radius=radius,
                                 method='niblack', offset=0.075)
    assert thresholded0.shape == image.shape


def test_threshold_2d_median(image):
    sigma = 2
    radius = 5
    thresholded = pre.threshold(image[:100, :100], sigma=sigma, radius=radius,
                                method='median')
    assert thresholded.shape == (100, 100)
    assert thresholded.dtype == bool


def test_threshold_2d_otsu(image):
    thresholded_otsu = pre.threshold(image)
    assert thresholded_otsu.shape == image.shape


def test_threshold_no_method(image):
    with pytest.raises(ValueError):
        pre.threshold(image, radius=1, method='no method')


def _total_variation(image):
    return sum(np.sum(np.abs(np.diff(image, axis=i)))
               for i in range(image.ndim))


def test_threshold_denoise(image):
    denoised_thresholded = pre.threshold(image, sigma=5, radius=15,
                                         smooth_method='tv')
    thresholded = pre.threshold(image, sigma=0, radius=15)
    assert (_total_variation(thresholded) >
            _total_variation(denoised_thresholded))
