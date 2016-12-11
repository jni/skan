import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, morphology
from skimage.color import gray2rgb
from .csr import summarise
from .pre import threshold


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
    """Overlay the skeleton pixels on the input image.

    Parameters
    ----------
    image : array, shape (M, N[, 3])
        The input image. Can be grayscale or RGB.
    skeleton : array, shape (M, N)
        The input 1-pixel-wide skeleton.

    Other Parameters
    ----------------
    image_cmap : matplotlib colormap name or object, optional
        If the input image is grayscale, colormap it with this colormap.
        The default is grayscale.
    color : tuple of float in [0, 1], optional
        The RGB color for the skeleton pixels.
    alpha : float, optional
        Blend the skeleton pixels with the given alpha.
    axes : matplotlib Axes
        The Axes on which to plot the image. If None, new ones are created.

    Returns
    -------
    axes : matplotlib Axes
        The Axis on which the image is drawn.
    """
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
    """Plot the image, and overlay the straight-line skeleton over it.

    Parameters
    ----------
    image : array, shape (M, N)
        The input image.
    skeleton : array, shape (M, N)
        A 1-pixel thick skeleton to overlay over `image`.

    Other Parameters
    ----------------
    image_cmap : matplotlib colormap name or object, optional
        The colormap to use for the input image. Defaults to grayscale.
    skeleton_color_source : string, optional
        The name of the column to use for the skeleton edge color. See the
        output of `skan.summarise` for valid choices. Most common choices
        would be:
        - skeleton-id: each individual skeleton (connected component) will
          have a different colour.
        - branch-type: each branch type (tip-tip, tip-junction,
          junction-junction, path-path). This is the default.
        - branch-distance: the curved length of the skeleton branch.
        - euclidean-distance: the straight-line length of the skeleton branch.
    skeleton_colormap : matplotlib colormap name or object, optional
        The colormap for the skeleton values.
    axes : matplotlib Axes object, optional
        An Axes object on which to draw. If `None`, a new one is created.

    Returns
    -------
    axes : matplotlib Axes object
        The Axes on which the plot is drawn.
    """
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


def pipeline_plot(image, *, sigma=0., radius=0, offset=0.,
                  figsize=(9, 9)):
    """Draw the image, the thresholded version, and its skeleton.

    Parameters
    ----------
    image : array, shape (M, N, ...[, 3])
        Input image, conformant with scikit-image data type
        specification [1]_.
    sigma : float, optional
        If positive, use Gaussian filtering to smooth the image before
        thresholding.
    radius : int, optional
        If given, use local median thresholding instead of global.
    offset : float, optional
        If given, reduce the threshold by this amount. Higher values
        result in more pixels above the threshold.
    figsize : 2-tuple of float, optional
        The width and height of the figure.

    Returns
    -------
    fig : matplotlib Figure
        The Figure containing all the plots
    axes : array of matplotlib Axes
        The four axes containing the drawn images.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = np.ravel(axes)
    axes[0].imshow(image)
    axes[0].axis('off')

    thresholded = threshold(image, sigma=sigma, radius=radius, offset=offset)
    axes[1].imshow(thresholded)
    axes[1].axis('off')

    skeleton = morphology.skeletonize(thresholded)
    overlay_skeleton_2d(image, skeleton, axes=axes[2])

    overlay_euclidean_skeleton_2d(image, skeleton, axes=axes[3])

    return fig, axes
