import numpy as np
from enum import Enum
from skimage.morphology import skeletonize
from skan import summarize, Skeleton # is this wrong?? should i not be importing from myself??
import napari
from magicgui.widgets import Container, ComboBox, PushButton, Label

class SkeletonizeMethod(Enum):
    Zhang = "zhang"
    Lee = "lee"


def skeletonize_labels(labels: "napari.types.LabelsData", method: SkeletonizeMethod) -> "napari.types.LabelsData":
    """Takes a labels layer and a skimage skeletonize method and generates a skeleton representation

    Parameters
    ----------
    labels : napari.types.LabelsData
        A labels layer containing data to skeletonize
    method : SkeletonizeMethod
        Enum denoting the chosen skeletonize method method

    Returns
    -------
    napari.types.LabelsData
        Labels layer depecting the extracted skeleton
    """
    binary_labels = (labels > 0).astype(np.uint8)
    # skeletonize returns a binary array, so we can just iltiply it with the labels to get appropriate colours
    skeletonized = skeletonize(binary_labels, method=method.value) * labels
    return skeletonized


def analyse_skeleton(labels: "napari.types.LabelsData"):
    # TODO: change 'analyse' to 'analyze' (and in napari.yaml)
    # TODO: change colour to color
    binary_labels = (labels > 0).astype(np.uint8)
    binary_skeleton = skeletonize(binary_labels)
    
    skeleton = skeletonize(binary_skeleton)
    paths_table = summarize(skeleton)
    colour_options = paths_table.features
    
class AnalyzeSkeletonWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Widget for 

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            napari viewer to add the widget to
        """
        super().__init__()
        self.viewer = viewer
        self.labels_combo = ComboBox(
                name='Labels Layer', choices=self.get_labels_layers
                )

        labels = viewer.layers[self.labels_combo.current_choice].data

        binary_labels = (labels > 0).astype(np.uint8)
        binary_skeleton = skeletonize(binary_labels)
        skeleton = Skeleton(binary_skeleton)
        # paths_table = summarize(binary_skeleton)
        all_paths = [skeleton.path_coordinates(i) 
                     for i in range(skeleton.n_paths)]

        paths_table = summarize(skeleton)
        self.features_combo = ComboBox(
                name='feature', choices=paths_table[:1]
                )
        self.features_combo.changed.connect(self.update_edge_color)
        self.skeleton_layer = viewer.add_shapes(
            all_paths,
            shape_type='path',
            properties=paths_table,
            edge_width=0.5,
            edge_color='skeleton-id',
            edge_colormap='tab10',
            )
        self.extend([self.labels_combo, self.features_combo])

    def update_edge_color(self, value):
        self.skeleton_layer.edge_color = value

    def get_labels_layers(self, combo):
        """Returns a list of existing labels to display

        Parameters
        ----------
        combo : magicgui ComboBox
            A dropdown to dispaly the layers

        Returns
        -------
        list[napari.layer.label]
            A list of curently existing layers
        """
        return [
                layer for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Labels)
                ]


if __name__ == "__main__":
    viewer = napari.Viewer()
    napari.run()
    

# TODO: make sure the widget doesnt crash when no data 