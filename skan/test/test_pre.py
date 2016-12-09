from scipy import special
import numpy as np
from hypothesis import given, strategies
from hypothesis.extra.numpy import arrays

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


uint8s = strategies.integers(min_value=0, max_value=255)


@given(image=arrays(dtype=np.uint8, shape=(15, 15)),
       sigma=strategies.integers(min_value=0, max_value=3),
       radius=strategies.integers(min_value=0, max_value=6))
def test_threshold2d(image, sigma, radius):
    radius = max(radius, 2 * sigma)
    thresholded0 = pre.threshold(image, sigma=sigma, radius=radius)
    assert thresholded0.dtype == bool
    assert thresholded0.shape == image.shape
    offset = np.random.randint(-15, 15)
    thresholded1 = pre.threshold(image, sigma=sigma, radius=radius,
                                 offset=offset)
    if offset > 0:
        assert np.all(thresholded1 <= thresholded0)
    else:
        assert np.all(thresholded1 >= thresholded0)
