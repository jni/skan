from magicgui import magicgui
import numpy as np
from enum import Enum
from skimage.morphology import skeletonize
from skan import summarize, Skeleton
from magicgui.widgets import Container, ComboBox, PushButton, Label, create_widget


class SkeletonizeMethod(Enum):
    zhang = "zhang"
    lee = "lee"

def get_skeleton(labels: "napari.layers.Labels", choice: SkeletonizeMethod) -> "napari.types.LayerDataTuple":
    binary_labels = (labels.data > 0).astype(np.uint8)
    binary_skeleton = skeletonize(binary_labels, method=choice.value)
    
    skeleton = Skeleton(binary_skeleton)

    all_paths = [skeleton.path_coordinates(i)  
            for i in range(skeleton.n_paths)]
    
    paths_table = summarize(skeleton)


    return (
        all_paths,
        {'shape_type': 'path', 'edge_colormap': 'tab10', 'metadata': {'skeleton': skeleton, 'features': paths_table}},
        'shapes',
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
        self.shapes_combo = create_widget(annotation = "napari.layers.Shapes", name = "skeleton")

        self.features_combo = ComboBox(
                name='feature', 
                )
        self.get_features_button = PushButton(name='Get features')
        self.get_features_button.clicked.connect(self.analyze_shapes_layer)

        self.extend([self.shapes_combo,self.features_combo, self.get_features_button])

    def analyze_shapes_layer(self, combo):
        """Perfom the analysis on the shape/skeleton layer and color the shape layer

        Parameters
        ----------
        combo : magicgui ComboBox
            A dropdown to dispaly the layers
        """
        current_layer = self.viewer.layers[self.shapes_combo.current_choice]
        current_layer.features = current_layer.metadata["features"]
        self.features_combo.choices = current_layer.features.columns

        self.features_combo.changed.connect(self.update_edge_color)

    def update_edge_color(self, value):
        """update the color of the currently selected shapes layer """        
        self.viewer.layers[self.shapes_combo.current_choice].edge_color = value


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()