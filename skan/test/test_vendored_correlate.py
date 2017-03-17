from time import time
from functools import reduce
import numpy as np
from skan.vendored import thresholding as th
from skimage.transform import integral_image
from scipy import ndimage as ndi


class Timer:
    def __init__(self):
        self.interval = 0

    def __enter__(self):
        self.t0 = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.interval = time() - self.t0


def test_fast_sauvola():
    image = np.random.rand(512, 512)
    w0 = 25
    w1 = 251
    _ = th.threshold_sauvola(image, window_size=3)
    with Timer() as t0:
        th.threshold_sauvola(image, window_size=w0)
    with Timer() as t1:
        th.threshold_sauvola(image, window_size=w1)
    assert t1.interval < 2 * t0.interval


def test_reference_correlation():
    ndim = 4
    shape = np.random.randint(2, 20, size=ndim)
    x = np.random.random(shape)
    kern = reduce(np.outer, [[-1, 0, 0, 1]] * ndim).reshape((4,) * ndim)
    px = np.pad(x, (2, 1), mode='reflect')
    pxi = integral_image(px)
    mean_fast = th.correlate_sparse(pxi, kern / 3 ** ndim, mode='valid')
    mean_ref = ndi.correlate(x, np.ones((3,) * ndim) / 3 ** ndim,
                             mode='mirror')
    np.testing.assert_allclose(mean_fast, mean_ref)
