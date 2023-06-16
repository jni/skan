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
    color_by_feature_widget.shapes_layer.changed.connect(
        lambda _: _update_feature_names(color_by_feature_widget)
    )
    _update_feature_names(color_by_feature_widget)

def _update_feature_names(color_by_feature_widget):
    shapes_layer = color_by_feature_widget.shapes_layer.value
    shapes_layer.features = shapes_layer.metadata["features"]
    color_by_feature_widget.feature_name.choices = shapes_layer.features.columns
    with color_by_feature_widget.feature_name.changed.blocked():
        color_by_feature_widget.feature_name.value = shapes_layer.features.columns[0]
        

@magic_factory(
        widget_init=populate_feature_choices,
        feature_name = {"widget_type": "ComboBox"}

)
def color_by_feature(shapes_layer:"napari.layers.Shapes", feature_name):
    current_column_type = shapes_layer.features[feature_name].dtype
    if current_column_type == "float64":
        shapes_layer.edge_colormap = CONTINUOUS_COLOR
    else:
        shapes_layer.edge_colormap = CAT_COLOR
    shapes_layer.edge_color = feature_name


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    data = np.zeros(dtype="int", shape=(10,10))
    data[1,:] = 1

    binary_labels = (data > 0).astype(np.uint8)
    binary_skeleton = skeletonize(binary_labels, method="zhang")
    
    skeleton = Skeleton(binary_skeleton)

    all_paths = [skeleton.path_coordinates(i)  
            for i in range(skeleton.n_paths)]
    
    paths_table = summarize(skeleton)
    skele = viewer.add_shapes(all_paths, shape_type="path", metadata= {'skeleton': skeleton, 'features': paths_table})
    napari.run()
