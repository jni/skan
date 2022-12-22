import numpy as np
from enum import Enum
from skimage.morphology import skeletonize

class SkeletonizeMethod(Enum):
    Zhang = "zhang"
    Lee = "lee"


def skeletonize_labels(labels: "napari.types.LabelsData", method: SkeletonizeMethod) -> "napari.types.LabelsData":
    binary_labels = (labels > 0).astype(np.uint8)
    skeletonized = skeletonize(binary_labels, method=method)
    return skeletonized
