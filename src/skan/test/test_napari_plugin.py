from skan.napari_skan import get_skeleton
from napari.layers import Labels
import numpy as np

def test_get_skeleton():
    label_data = np.zeroes(shape=(10,10), dtype=int)
    labels_layer = Labels(label_data)
    skeleton_type = "zhang"

    