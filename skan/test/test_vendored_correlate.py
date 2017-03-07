from contextlib import contextmanager
from time import time
import numpy as np
from skan.vendored import thresholding as th


@contextmanager
def timer():
    result = [0.]
    t = time()
    yield result
    result[0] = time() - t


def test_fast_sauvola():
    image = np.random.rand(512, 512)
    w0 = 25
    w1 = 251
    _ = th.threshold_sauvola(image, window_size=3)
    with timer() as t0:
        th.threshold_sauvola(image, window_size=w0)
    with timer() as t1:
        th.threshold_sauvola(image, window_size=w1)
    assert t1[0] < 2 * t0[0]
