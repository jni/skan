from magicgui import magicgui, magic_factory
import numpy as np
from enum import Enum
from skimage.morphology import skeletonize
from skan import summarize, Skeleton
from magicgui.widgets import Container, ComboBox, PushButton, Label, create_widget


CAT_COLOR = "tab10"
CONTINUOUS_COLOR = "viridis"
class SkeletonizeMethod(Enum):
    zhang = "zhang"
    lee = "lee"

def get_skeleton(labels: "napari.layers.Labels", choice: SkeletonizeMethod) -> "napari.types.LayerDataTuple":
    """Takes in a napari shapes layer and a skeletonization method (for skan.morphology),
    genertates a skeleton structure and places it on a shapes layer

    Parameters
    ----------
    labels : napari.layers.Labels
        Labels layer containing data to skeletonize
    choice : SkeletonizeMethod
        Enum containing string corresponding to skeletonization method

    Returns
    -------
    napari.types.LayerDataTuple
        Layer data with skeleton
    """
    binary_labels = (labels.data > 0).astype(np.uint8)
    binary_skeleton = skeletonize(binary_labels, method=choice.value)
    
    skeleton = Skeleton(binary_skeleton)

    all_paths = [skeleton.path_coordinates(i)  
            for i in range(skeleton.n_paths)]
    
    paths_table = summarize(skeleton) # option to have main_path = True (or something) changing header

    return (
        all_paths,
        {'shape_type': 'path', 'edge_colormap': 'tab10', 'metadata': {'skeleton': skeleton, 'features': paths_table}},
        'shapes',
        )


def populate_feature_choices(color_by_feature_widget):
    """Runs on widget init, connects combobox to function to update
    combobox options

    Parameters
    ----------
    color_by_feature_widget : _type_
        _description_
    """
    color_by_feature_widget.shapes_layer.changed.connect(
        lambda _: _update_feature_names(color_by_feature_widget)
    )
    _update_feature_names(color_by_feature_widget)

def _update_feature_names(color_by_feature_widget):
    """Search for a shapes layer with approptiate metadata for skeletons

    Parameters
    ----------
    color_by_feature_widget : magicgui Widget
        widget that contains reference to shapes layers
    """
    shapes_layer = color_by_feature_widget.shapes_layer.value
    if shapes_layer.features.empty and "features" in shapes_layer.metadata:
        shapes_layer.features = shapes_layer.metadata["features"]

    def get_choices(features_combo):
        return shapes_layer.features.columns

    color_by_feature_widget.feature_name.choices = get_choices
        

@magic_factory(
        widget_init=populate_feature_choices,
        feature_name = {"widget_type": "ComboBox"}
)
def color_by_feature(shapes_layer:"napari.layers.Shapes", feature_name):
    """Check the currently selected feature and update edge colors
    Can be any color from matplotlib.colormap

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The shapes layer currently selected in the Layers combobox
    feature_name : String
        The feature name currently selected in the features combobox
    """
    current_column_type = shapes_layer.features[feature_name].dtype
    if current_column_type == "float64":
        shapes_layer.edge_colormap = CONTINUOUS_COLOR
    else:
        shapes_layer.edge_colormap = CAT_COLOR
    shapes_layer.edge_color = feature_name
