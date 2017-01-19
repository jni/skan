import os
import imageio
from scipy import special
import numpy as np
from hypothesis import given, strategies
import pytest

from skan import pre


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


def test_threshold2d():
    rundir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(rundir, 'data')
    image = imageio.imread(os.path.join(datadir, 'retic.tif'), format='fei')
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

    thresholded2 = pre.threshold(image, sigma=sigma, radius=radius,
                                 method='niblack')
    thresholded3 = pre.threshold(image[:250, :250], sigma=sigma,
                                 radius=radius, method='median')
