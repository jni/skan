from skan.napari_skan import labels_to_skeleton_shapes, _update_feature_names
from skimage import data, morphology
from napari.layers import Labels
import numpy as np
from skan.napari_skan import SkeletonizeMethod
from skan.csr import Skeleton
import pandas as pd
import napari


def make_trivial_labels_layer():
    label_data = np.zeros(shape=(10, 10), dtype=int)
    label_data[5, 1:9] = 1
    labels_layer = Labels(label_data)
    return labels_layer


def test_get_skeleton_simple():
    labels_layer = make_trivial_labels_layer()
    skeleton_type = SkeletonizeMethod.zhang
    shapes_data, layer_kwargs, _ = labels_to_skeleton_shapes(
            labels_layer, skeleton_type
            )

    assert type(layer_kwargs['metadata']['skeleton']) is Skeleton
    np.testing.assert_array_equal(
            shapes_data[0],
            [[5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8]]
            )
    assert len(shapes_data) == 1
    assert 'features' in layer_kwargs
    assert type(layer_kwargs['features']) is pd.DataFrame


def test_get_skeleton_horse():
    horse = np.logical_not(data.horse().astype(bool))
    labels_layer = Labels(horse)
    skeleton_type = SkeletonizeMethod.zhang
    shapes_data, layer_kwargs, _ = labels_to_skeleton_shapes(
            labels_layer, skeleton_type
            )
    assert len(shapes_data) == 24  # 24 line segments in the horse skeleton
    assert 'features' in layer_kwargs
    assert type(layer_kwargs['features']) is pd.DataFrame


def test_gui(make_napari_viewer):
    viewer = make_napari_viewer()
    horse = np.logical_not(data.horse().astype(bool))

    labels_layer = viewer.add_labels(horse)

    ldt = labels_to_skeleton_shapes(labels_layer, SkeletonizeMethod.zhang)
    (skel_layer,) = viewer._add_layer_from_data(*ldt)

    dw, widget = viewer.window.add_plugin_dock_widget(
            'skan', 'Color Skeleton Widget'
            )
    widget.feature_name.value = 'euclidean_distance'
    widget()
    layer = viewer.layers[-1]
    assert layer.edge_colormap.name == 'viridis'
    assert len(layer.data) == 24
