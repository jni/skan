import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
import networkx as nx
from skimage import img_as_float, morphology
from skimage.color import gray2rgb


def _normalise_image(image, *, image_cmap=None):
    image = img_as_float(image)
    if image.ndim == 2:
        if image_cmap is None:
            image = gray2rgb(image)
        else:
            image = plt.get_cmap(image_cmap)(image)[..., :3]
    return image


def pixel_perfect_figsize(image, dpi=80):
    """Return the Matplotlib figure size tuple (w, h) for given image and dpi.

    Parameters
    ----------
    image : array, shape (M, N[, 3])
        The image to be plotted.
    dpi : int, optional
        The desired figure dpi.

    Returns
    -------
    figsize : tuple of float
        The desired figure size.

    Examples
    --------
    >>> image = np.empty((768, 1024))
    >>> pixel_perfect_figsize(image)
    (12.8, 9.6)
    """
    hpix, wpix = image.shape[:2]
    return wpix/dpi, hpix/dpi


def overlay_skeleton_2d(image, skeleton, *,
                        image_cmap=None, color=(1, 0, 0), alpha=1,
                        dilate=0, axes=None):
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
    dilate : int, optional
        Dilate the skeleton by this amount. This is useful when rendering
        large images where aliasing may cause some pixels of the skeleton
        not to be drawn.
    axes : matplotlib Axes
        The Axes on which to plot the image. If None, new ones are created.

    Returns
    -------
    axes : matplotlib Axes
        The Axis on which the image is drawn.
    """
    image = _normalise_image(image, image_cmap=image_cmap)
    skeleton = skeleton.astype(bool)
    if dilate > 0:
        selem = morphology.disk(dilate)
        skeleton = morphology.binary_dilation(skeleton, selem)
    if axes is None:
        fig, axes = plt.subplots()
    image[skeleton] = alpha * np.array(color) + (1 - alpha) * image[skeleton]
    axes.imshow(image)
    axes.axis('off')
    return axes


def overlay_euclidean_skeleton_2d(image, stats, *,
                                  image_cmap=None,
                                  skeleton_color_source='branch-type',
                                  skeleton_colormap='viridis',
                                  axes=None):
    """Plot the image, and overlay the straight-line skeleton over it.

    Parameters
    ----------
    image : array, shape (M, N)
        The input image.
    stats : array, shape (M, N)
        Skeleton statistics.

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
    summary = stats
    # transforming from row, col to x, y
    coords_cols = (['image-coord-src-%i' % i for i in [1, 0]] +
                   ['image-coord-dst-%i' % i for i in [1, 0]])
    coords = summary[coords_cols].values.reshape((-1, 2, 2))
    if axes is None:
        fig, axes = plt.subplots()
    axes.imshow(image)
    axes.axis('off')
    color_values = summary[skeleton_color_source]
    cmap = plt.get_cmap(skeleton_colormap,
                        min(len(np.unique(color_values)), 256))
    colormapped = cmap((color_values - np.min(color_values)) /
                       (np.max(color_values) - np.min(color_values)))
    linecoll = collections.LineCollection(coords, colors=colormapped)
    axes.add_collection(linecoll)
    return axes


def overlay_skeleton_2d_class(skeleton, *,
                              image_cmap='gray',
                              skeleton_color_source='path_means',
                              skeleton_colormap='viridis',
                              vmin=None, vmax=None,
                              axes=None):
    """Plot the image, and overlay the skeleton over it.

    Parameters
    ----------
    skeleton : skan.Skeleton object
        The input skeleton, which contains both the skeleton and the source
        image.

    Other Parameters
    ----------------
    image_cmap : matplotlib colormap name or object, optional
        The colormap to use for the input image. Defaults to grayscale.
    skeleton_color_source : string or callable, optional
        The name of the method to use for the skeleton edge color. See the
        documentation of `skan.Skeleton` for valid choices. Most common choices
        would be:

        - path_means: the mean value of the skeleton along each path.
        - path_lengths: the length of each path.
        - path_stdev: the standard deviation of pixel values along the path.

        Alternatively, a callable can be provided that takes as input a
        Skeleton object and outputs a list of floating point values of the same
        length as the number of paths.

    skeleton_colormap : matplotlib colormap name or object, optional
        The colormap for the skeleton values.
    vmin, vmax : float, optional
        The minimum and maximum values for the colormap. Use this to pin the
        colormapped values to a certain range.
    axes : matplotlib Axes object, optional
        An Axes object on which to draw. If `None`, a new one is created.

    Returns
    -------
    axes : matplotlib Axes object
        The Axes on which the plot is drawn.
    mappable : matplotlib ScalarMappable object
        The mappable values corresponding to the line colors. This can be used
        to create a colorbar for the plot.
    """
    image = skeleton.source_image
    if axes is None:
        fig, axes = plt.subplots()
    axes.imshow(image, cmap=image_cmap)
    if callable(skeleton_color_source):
        values = skeleton_color_source(skeleton)
    elif hasattr(skeleton, skeleton_color_source):
        values = getattr(skeleton, skeleton_color_source)()
    else:
        raise ValueError('Unknown skeleton color source: %s. Provide an '
                         'attribute of skan.csr.Skeleton or a callable.' %
                         skeleton_color_source)
    cmap = plt.get_cmap(skeleton_colormap,
                        min(len(np.unique(values)), 256))
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    mapping_values = (values - vmin) / (vmax - vmin)
    mappable = plt.cm.ScalarMappable(plt.Normalize(vmin, vmax), cmap)
    mappable._A = mapping_values
    colors = cmap(mapping_values)
    coordinates = [skeleton.path_coordinates(i)[:, ::-1]
                   for i in range(skeleton.n_paths)]
    linecoll = collections.LineCollection(coordinates, colors=colors)
    axes.add_collection(linecoll)
    return axes, mappable


def pipeline_plot(image, thresholded, skeleton, stats, *,
                  figure=None, axes=None, figsize=(9, 9)):
    """Draw the image, the thresholded version, and its skeleton.

    Parameters
    ----------
    image : array, shape (M, N, ...[, 3])
        Input image, conformant with scikit-image data type
        specification [1]_.
    thresholded : array, same shape as image
        Binarized version of the input image.
    skeleton : array, same shape as image
        Skeletonized version of the input image.
    stats : pandas DataFrame
        Skeleton statistics from the input image/skeleton.

    Other Parameters
    ----------------
    figure : matplotlib Figure, optional
        If given, where to make the plots.
    axes : array of matplotlib Axes, optional
        If given, use these axes to draw the plots. Should have len 4.
    figsize : 2-tuple of float, optional
        The width and height of the figure.
    smooth_method : {'Gaussian', 'TV', 'NL'}, optional
        Which denoising method to use on the image.

    Returns
    -------
    fig : matplotlib Figure
        The Figure containing all the plots
    axes : array of matplotlib Axes
        The four axes containing the drawn images.

    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/user_guide/data_types.html
    """
    if figure is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize,
                                 sharex=True, sharey=True)
        axes = np.ravel(axes)
    else:
        fig = figure
        if axes is None:
            ax0 = fig.add_subplot(2, 2, 1)
            axes = [ax0] + [fig.add_subplot(2, 2, i, sharex=ax0, sharey=ax0)
                            for i in range(2, 5)]

    axes = np.ravel(axes)
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(thresholded, cmap='gray')
    axes[1].axis('off')

    overlay_skeleton_2d(image, skeleton, axes=axes[2])

    overlay_euclidean_skeleton_2d(image, stats, axes=axes[3])

    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    return fig, axes


def _clean_positions_dict(d, g):
    for k in list(d.keys()):
        if k not in g:
            del d[k]
        elif g.degree(k) == 0:
            g.remove_node(k)


def overlay_skeleton_networkx(csr_graph, coordinates, *, axis=None,
                              image=None, cmap=None, **kwargs):
    """Draw the skeleton as a NetworkX graph, optionally overlaid on an image.

    Due to the size of NetworkX drawing elements, this is only recommended
    for very small skeletons.

    Parameters
    ----------
    csr_graph : SciPy Sparse matrix
        The skeleton graph in SciPy CSR format.
    coordinates : array, shape (N_points, 2)
        The coordinates of each point in the skeleton. ``coordinates.shape[0]``
        should be equal to ``csr_graph.shape[0]``.

    Other Parameters
    ----------------
    axis : Matplotlib Axes object, optional
        The Axes on which to plot the data. If None, a new figure and axes will
        be created.
    image : array, shape (M, N[, 3])
        An image on which to overlay the skeleton. ``image.shape`` should be
        greater than ``np.max(coordinates, axis=0)``.
    **kwargs : keyword arguments
        Arguments passed on to `nx.draw_networkx`. Particularly useful ones
        include ``node_size=`` and ``font_size=``.
    """
    if axis is None:
        _, axis = plt.subplots()
    if image is not None:
        cmap = cmap or 'gray'
        axis.imshow(image, cmap=cmap)
    gnx = nx.from_scipy_sparse_matrix(csr_graph)
    # Note: we invert the positions because Matplotlib uses x/y for
    # scatterplot, but the coordinates are row/column NumPy indexing
    positions = dict(zip(range(coordinates.shape[0]), coordinates[:, ::-1]))
    _clean_positions_dict(positions, gnx)  # remove nodes not in Graph
    nx.draw_networkx(gnx, pos=positions, ax=axis, **kwargs)
    return axis
