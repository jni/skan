import dask.array as da
import pytest
import numpy as np

from skan import Skeleton, summarize
from skan.dask_compat import DaskSkeleton
from skan._testdata import skeleton0

pytest.importorskip("dask_image")


@pytest.fixture
def test_skeleton_data():
    dask_skeleton_image = da.block([skeleton0, skeleton0])
    numpy_skeleton_image = np.array(dask_skeleton_image)
    return dask_skeleton_image, numpy_skeleton_image


def test_dask_compat_summarize(test_skeleton_data):
    dask_skeleton_image, numpy_skeleton_image = test_skeleton_data
    dask_skel = DaskSkeleton(dask_skeleton_image)
    numpy_skel = Skeleton(numpy_skeleton_image)
    result_dask = summarize(dask_skel)
    result_numpy = summarize(numpy_skel)
    assert len(result_dask) == len(result_numpy)
    assert result_dask.equals(result_numpy)
