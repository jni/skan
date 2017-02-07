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
    data = pipe.process_images([image_filename], 'fei', 5e-8, 0.1, 0.075,
                               'Scan/PixelHeight')
    assert type(data[0]) == pandas.DataFrame
    assert data[0].shape[0] > 0


def test_pipe_figure(image_filename):
    with tempfile.TemporaryDirectory() as tempdir:
        data = pipe.process_images([image_filename], 'fei', 5e-8, 0.1, 0.075,
                                   'Scan/PixelHeight',
                                   save_skeleton='skeleton-plot-',
                                   output_folder=tempdir)
        expected_output = os.path.join(tempdir, 'skeleton-plot-' +
                                       os.path.basename(image_filename)[:-4] +
                                       '.png')
        assert os.path.exists(expected_output)
    assert type(data[0]) == pandas.DataFrame
    assert data[0].shape[0] > 0
