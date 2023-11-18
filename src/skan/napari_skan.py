from magicgui import magic_factory
import numpy as np
from enum import Enum
from skimage.morphology import skeletonize
from skan import summarize, Skeleton

CAT_COLOR = "tab10"
CONTINUOUS_COLOR = "viridis"


class SkeletonizeMethod(Enum):
    """Use enum for method choice for easier use with magicgui."""
    zhang = "zhang"
    lee = "lee"


def labels_to_skeleton_shapes(
        labels: "napari.layers.Labels", choice: SkeletonizeMethod
        ) -> "napari.types.LayerDataTuple":
    """Skeletonize a labels layer using given method and export as Shapes.

    Parameters
    ----------
    labels : napari.layers.Labels
        Labels layer containing data to skeletonize
    choice : SkeletonizeMethod
        Enum corresponding to skeletonization method

    Returns
    -------
    napari.types.LayerDataTuple
        Shapes layer data with skeleton
    """
    binary_labels = (labels.data > 0).astype(np.uint8)
    binary_skeleton = skeletonize(binary_labels, method=choice.value)

    skeleton = Skeleton(binary_skeleton)

    all_paths = [skeleton.path_coordinates(i) for i in range(skeleton.n_paths)]

    # option to have main_path = True (or something) changing header
    paths_table = summarize(skeleton, separator='_')
    layer_kwargs = {
            'shape_type': 'path',
            'edge_colormap': 'tab10',
            'features': paths_table,
            'metadata': {'skeleton': skeleton},
            }

    return all_paths, layer_kwargs, 'shapes'


def _populate_feature_choices(color_by_feature_widget):
    """Update feature names combobox when source layer is changed.

    This runs on widget initialization and on every change of Shapes layer
    thereafter.

    Parameters
    ----------
    color_by_feature_widget : function that takes widget as input
        Function that takes in the widget and modifies the choices in-place.
    """
    color_by_feature_widget.shapes_layer.changed.connect(
            lambda _: _update_feature_names(color_by_feature_widget)
            )
    _update_feature_names(color_by_feature_widget)


def _update_feature_names(color_by_feature_widget):
    """Search for a shapes layer with appropriate metadata for skeletons

    Parameters
    ----------
    color_by_feature_widget : magicgui Widget
        widget that contains reference to shapes layers
    """
    shapes_layer = color_by_feature_widget.shapes_layer.value

    def get_choices(features_combo):
        """Closure to use the current shapes layer to update given combobox."""
        return shapes_layer.features.columns

    color_by_feature_widget.feature_name.choices = get_choices


@magic_factory(
        widget_init=_populate_feature_choices,
        feature_name={"widget_type": "ComboBox"}
        )
def color_by_feature(shapes_layer: "napari.layers.Shapes", feature_name):
    """Check the currently selected feature and update edge colors

    TODO: allow selecting a colormap.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        A napari Shapes layer.
    feature_name : String
        A string corresponding to a feature column present in the Shapes layer.
    """
    current_column_type = shapes_layer.features[feature_name].dtype
    if current_column_type == "float64":
        shapes_layer.edge_colormap = CONTINUOUS_COLOR
    else:
        shapes_layer.edge_colormap = CAT_COLOR
    shapes_layer.edge_color = feature_name
