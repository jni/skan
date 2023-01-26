import numpy as np
from enum import Enum
from skimage.morphology import skeletonize

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
    skeletonized = skeletonize(binary_labels, method=method.value)
    return skeletonized
