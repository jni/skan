import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.color import gray2rgb
from .csr import summarise


def _normalise_image(image, *, image_cmap=None):
    image = img_as_float(image)
    if image.ndim == 2:
        if image_cmap is None:
            image = gray2rgb(image)
        else:
            image = plt.get_cmap(image_cmap)(image)[..., :3]
    return image


def overlay_skeleton_2d(image, skeleton, *,
                        image_cmap=None, color=(1, 0, 0), alpha=1, axes=None):
    image = _normalise_image(image, image_cmap=image_cmap)
    skeleton = skeleton.astype(bool)
    if axes is None:
        fig, axes = plt.subplots()
    image[skeleton] = alpha * np.array(color) + (1 - alpha) * image[skeleton]
    axes.imshow(image)
    axes.axis('off')
    return axes


def overlay_euclidean_skeleton_2d(image, skeleton, *,
                                  image_cmap=None,
                                  skeleton_color_source='branch-type',
                                  skeleton_colormap='viridis',
                                  axes=None):
    image = _normalise_image(image, image_cmap=image_cmap)
    summary = summarise(skeleton)
    coords_cols = (['img-coord-0-%i' % i for i in range(2)] +
                   ['img-coord-1-%i' % i for i in range(2)])
    coords = summary[coords_cols]
    if axes is None:
        fig, axes = plt.subplots()
    axes.imshow(image)
    axes.axis('off')
    color_values = summary[skeleton_color_source]
    cmap = plt.get_cmap(skeleton_colormap,
                        min(len(np.unique(color_values)), 256))
    colormapped = cmap((color_values - np.min(color_values)) /
                       (np.max(color_values) - np.min(color_values)))
    for ((_, (r0, c0, r1, c1)), color) in zip(coords.iterrows(),
                                              colormapped):
        axes.plot([c0, c1], [r0, r1], color=color, marker=None)
    return axes
