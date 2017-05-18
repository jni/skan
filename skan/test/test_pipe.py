import os
import pytest
import tempfile

import pandas
from skan import pipe

@pytest.fixture
def image_filename():
    rundir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(rundir, 'data')
    return os.path.join(datadir, 'retic.tif')


def test_pipe(image_filename):
    results = pipe.process_images([image_filename], 'fei', 5e-8, 0.1, 0.075,
                                  'Scan/PixelHeight')
    single_image_data, data = results
    assert type(data[0]) == pandas.DataFrame
    assert data[0].shape[0] > 0

    results2 = pipe.process_images([image_filename], 'fei', 5e-8, 0.1, 0.075,
                                   'Scan/PixelHeight', crop_radius=75)
    single_image_data2, data2 = results2
    assert (single_image_data[1].shape[0] ==
            single_image_data2[1].shape[0] + 150)
    assert data2[0].shape[0] < data[0].shape[0]
