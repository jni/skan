from scipy import special
import numpy as np
from hypothesis import given, strategies
from hypothesis.extra.numpy import arrays
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


uint8s = strategies.integers(min_value=0, max_value=255)
methods = [strategies.just(v) for v in ['sauvola', 'niblack', 'median']]


@pytest.mark.skip
@given(image=arrays(dtype=np.uint8, shape=(15, 15)),
       sigma=strategies.integers(min_value=0, max_value=3),
       radius=strategies.integers(min_value=0, max_value=6),
       method=strategies.one_of(methods))
def test_threshold2d(image, sigma, radius, method):
    if np.all(image == image[0, 0]):
        return
    radius = max(radius, 2 * sigma)
    thresholded0 = pre.threshold(image, sigma=sigma, radius=radius,
                                 method=method)
    assert thresholded0.dtype == bool
    assert thresholded0.shape == image.shape
    offset = 0.025 + np.random.rand()
    thresholded1 = pre.threshold(image, sigma=sigma, radius=radius,
                                 method=method, offset=offset)
    if method == 'median':
        if offset > 0:
            assert np.all(thresholded1 <= thresholded0)
        else:
            assert np.all(thresholded1 >= thresholded0)
    else:
        if offset > 0.2:
            assert np.all(thresholded1 >= thresholded0)
        else:
            assert np.all(thresholded1 <= thresholded0)
