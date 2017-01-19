import os
import pytest

import pandas
from skan import pipe

@pytest.fixture
def image_filename():
    rundir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(rundir, 'data')
    return os.path.join(datadir, 'retic.tif')


def test_pipe(image_filename):
    data = pipe.process_images([image_filename], 'fei', 5e-8, 0.1, 0.075,
                               'Scan/PixelHeight')
    assert type(data) == pandas.DataFrame
    assert data.shape[0] > 0
