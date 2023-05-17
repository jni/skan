import numpy as np
from enum import Enum
from skimage.morphology import skeletonize
from skan import summarize, Skeleton
import napari
from magicgui.widgets import Container, ComboBox, PushButton, Label, create_widget
from napari.layers import Shapes 
class SkeletonizeMethod(Enum):
    zhang = "zhang"
    lee = "lee"

def get_skeleton(labels: "napari.layers.Labels", choice: SkeletonizeMethod) -> "napari.layers.Shapes":
    binary_labels = (labels.data > 0).astype(np.uint8)
    binary_skeleton = skeletonize(binary_labels, method=choice.value)
    
    skeleton = Skeleton(binary_skeleton)

    all_paths = [skeleton.path_coordinates(i)  
            for i in range(skeleton.n_paths)]

    return Shapes(all_paths,
        shape_type='path',
        edge_colormap='tab10',
        metadata={"skeleton": skeleton}
        )

class AnalyseSkeleton(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Widget for analyzing skeleton of an existing shapes layer 

        Parameters 
        ----------
        viewer : napari.viewer.Viewer
            napari viewer to add the widget to
        """
        super().__init__()
        self.viewer = viewer
        # self.shapes_combo = ComboBox(
        #     name='Shapes Layer', 
        #     choices=self.get_shapes_layers
        # )
        self.shapes_combo = create_widget(annotation = Shapes)
        self.analyze_button = PushButton(name='Analyze')
        self.analyze_button.clicked.connect(self.analyze_shapes_layer)

        self.extend([self.shapes_combo,self.analyze_button])

    def analyze_shapes_layer(self, combo):
        """Perfom the analysis on the shape/skeleton layer and color the shape layer

        Parameters
        ----------
        combo : magicgui ComboBox
            A dropdown to dispaly the layers
        """
        paths_table = summarize(self.viewer.layers[self.shapes_combo.current_choice].metadata["skeleton"])
        self.viewer.layers[self.shapes_combo.current_choice].features = paths_table
        self.features_combo = ComboBox(
                name='feature', 
                choices=paths_table[:1]
                )

        self.features_combo.changed.connect(self.update_edge_color)

        self.extend([self.features_combo])

    def get_shapes_layers(self, combo):
        """return a list of shapes layers"""
        return [
                layer for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Shapes)
                ]

    def update_edge_color(self, value):
        """update the color of the currently selected shapes layer """        
        self.viewer.layers[self.shapes_combo.current_choice].edge_color = value

if __name__ == "__main__":
    viewer = napari.Viewer()
    napari.run()