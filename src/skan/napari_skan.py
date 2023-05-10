import numpy as np
from enum import Enum
from skimage.morphology import skeletonize
from skan import summarize, Skeleton
import napari
from magicgui.widgets import Container, ComboBox, PushButton, Label

class SkeletonizeMethod(Enum):
    zhang = "zhang"
    lee = "lee"


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

    skeletonized = skeletonize(binary_labels, method=method.value) * labels
    return skeletonized


def analyze_skeleton(labels: "napari.types.LabelsData"):
    binary_labels = (labels > 0).astype(np.uint8)
    binary_skeleton = skeletonize(binary_labels)
    
    skeleton = skeletonize(binary_skeleton)
    paths_table = summarize(skeleton)
    color_options = paths_table.features
    
class SkeletonizeWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Widget for skeletonizing labels layer. 

        Parameters 
        ----------
        viewer : napari.viewer.Viewer
            napari viewer to add the widget to
        """
        super().__init__()
        self.viewer = viewer
        self.current_label_layer = None
        self.labels_combo = ComboBox(
                name='Labels Layer', choices=self.get_labels_layers
                )
        self.methods_combo = ComboBox(
                name="Method", choices = SkeletonizeMethod
        )
        self.skeletonize_button = PushButton(name='Skeletonize')
        self.skeletonize_button.clicked.connect(self.make_skeleton_layer)
        
        self.extend([self.labels_combo, self.methods_combo, self.skeletonize_button])

    def make_skeleton_layer(self):
        layer_name = self.labels_combo.current_choice
        self.current_label_layer = self.viewer.layers[layer_name].data
        binary_labels = (self.current_label_layer > 0).astype(np.uint8)
        binary_skeleton = skeletonize(binary_labels, method=self.methods_combo.current_choice)
        
        skeleton = Skeleton(binary_skeleton)

        all_paths = [skeleton.path_coordinates(i)  
                for i in range(skeleton.n_paths)]
        self.skeleton_layer = self.viewer.add_shapes(
            all_paths,
            shape_type='path',
            # features=paths_table,
            # edge_width=0.5,
            # edge_color='skeleton-id',
            edge_colormap='tab10',
            metadata={"skeleton": skeleton}
        )

    def set_current_layer(self):
        #TODO: error test
        self.current_label_layer = self.viewer.layers[self.labels_combo.current_choice].data

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


class AnalyseSkeleton(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Widget for skeletonizing labels layer. 

        Parameters 
        ----------
        viewer : napari.viewer.Viewer
            napari viewer to add the widget to
        """
        super().__init__()
        self.viewer = viewer
        self.shapes_combo = ComboBox(
            name='Shapes Layer', 
            choices=self.get_shapes_layers
        )
        self.analyze_button = PushButton(name='Analyze')
        self.analyze_button.clicked.connect(self.analyze_shapes_layer)

        self.extend([self.shapes_combo,self.analyze_button])

    def analyze_shapes_layer(self, combo):
        paths_table = summarize(self.viewer.layers[self.shapes_combo.current_choice].metadata["skeleton"])
        self.viewer.layers[self.shapes_combo.current_choice].features = paths_table
        self.features_combo = ComboBox(
                name='feature', 
                choices=paths_table[:1]
                )
        
        #TODO: upadte shapes parameters

        self.features_combo.changed.connect(self.update_edge_color)

        self.extend([self.features_combo])

    def get_shapes_layers(self, combo):
        return [
                layer for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Shapes)
                ]

    def update_edge_color(self, value):
        print(value)
        
        self.viewer.layers[self.shapes_combo.current_choice].edge_color = value

if __name__ == "__main__":
    viewer = napari.Viewer()
    napari.run()