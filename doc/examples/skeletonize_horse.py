import napari
from skimage import data, morphology
from skan import Skeleton
from skan.napari_skan import labels_to_skeleton_shapes, SkeletonizeMethod
import numpy as np

viewer = napari.Viewer()
horse = np.logical_not(data.horse().astype(bool))

labels_layer = viewer.add_labels(horse)

ldt = labels_to_skeleton_shapes(labels_layer, SkeletonizeMethod.zhang)
(skel_layer,) = viewer._add_layer_from_data(*ldt)

dw, widget = viewer.window.add_plugin_dock_widget(
        'skan', 'Color Skeleton Widget'
        )

napari.run()
