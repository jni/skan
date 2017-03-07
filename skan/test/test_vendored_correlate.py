from time import time
import numpy as np
from skan.vendored import thresholding as th


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
