from skan.napari_skan import get_skeleton
from napari.layers import Labels
import numpy as np
from skan.napari_skan import SkeletonizeMethod
from skan.csr import Skeleton as skan_skeleton
import pandas as pd


def make_trivial_labels_layer():
    label_data = np.zeros(shape=(10,10), dtype=int)
    label_data[5,1:9] = 1
    labels_layer = Labels(label_data)
    return labels_layer


def test_get_skeleton_simple():
    labels_layer = make_trivial_labels_layer()
    skeleton_type = SkeletonizeMethod.zhang
    skeleton = get_skeleton(labels_layer, skeleton_type)

    assert type(skeleton[1]["metadata"]["skeleton"]) is skan_skeleton
    np.testing.assert_array_equal(skeleton[0][0], [[5,1], [5,2], [5,3], [5,4], [5,5], [5,6], [5,7], [5,8]])
    assert len(skeleton[0]) == 1
    assert type(skeleton[1]) is dict
    assert len(skeleton[1]) == 3
    assert type(skeleton[1]["metadata"]["features"]) is pd.DataFrame


def test_get_skeleton_horse():
    from skimage import data
    horse = np.logical_not(data.horse().astype(bool))
    labels_layer = Labels(horse)
    skeleton_type = SkeletonizeMethod.zhang
    skeleton = get_skeleton(labels_layer, skeleton_type)
    assert len(skeleton[0]) == 24
    assert type(skeleton[1]) is dict
    assert len(skeleton[1]) == 3
    assert type(skeleton[1]["metadata"]["features"]) is pd.DataFrame

test_get_skeleton_simple()
test_get_skeleton_horse()
