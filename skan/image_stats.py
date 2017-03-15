import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import csr


def mesh_sizes(skeleton):
    """Compute the area in pixels of the spaces between skeleton branches.

    This only makes sense for 2D images.

    Parameters
    ----------
    skeleton : array, shape (M, N)
        An image containing a single-pixel-wide closed skeleton.

    Returns
    -------
    sizes : array of int, shape (P,)
        The sizes of all spaces delineated by the skeleton *not* touching
        the borders.

    Examples
    --------
    >>> image = np.array([[0, 0, 1, 0, 0],
    ...                   [0, 0, 1, 1, 1],
    ...                   [0, 0, 1, 0, 0],
    ...                   [0, 1, 0, 1, 0]])
    >>> print(mesh_sizes(image))
    []
    >>> from skan.nputil import pad
    >>> image2 = pad(image, 1)  # make sure mesh not touching border
    >>> print(mesh_sizes(image2))  # sizes in row order of first pixel in space
    [7 2 3 1]
    """
    spaces = ~skeleton.astype(bool)
    labeled = ndi.label(spaces)[0]
    touching_border = np.unique(np.concatenate((labeled[0], labeled[-1],
                                                labeled[:, 0],
                                                labeled[:, -1])))
    sizes = np.bincount(labeled.flat)
    sizes[touching_border] = 0
    sizes = sizes[sizes != 0]
    return sizes


def image_summary(skeleton, *, spacing=1):
    """Compute some summary statistics for an image.

    Parameters
    ----------
    skeleton : array, shape (M, N)
        The input image.

    Other Parameters
    ----------------
    spacing : float or array of float, shape (`skeleton.ndim`,)
        The resolution along each axis of `skeleton`.

    Returns
    -------
    stats : pandas.DataFrame
        Selected statistics about the image.
    """
    stats = pd.DataFrame()
    stats['scale'] = [spacing]
    g, coords, degimg = csr.skeleton_to_csgraph(skeleton, spacing=spacing)
    degrees = np.diff(g.indptr)
    num_junctions = np.sum(degrees > 2)
    stats['number of junctions'] = num_junctions
    pixel_area = (spacing ** skeleton.ndim if np.isscalar(spacing) else
                  np.prod(spacing))
    stats['area'] = np.prod(skeleton.shape) * pixel_area
    stats['junctions per unit area'] = (stats['number of junctions'] /
                                        stats['area'])
    sizes = mesh_sizes(skeleton)
    stats['average mesh area'] = np.mean(sizes)
    stats['median mesh area'] = np.median(sizes)
    stats['mesh area standard deviation'] = np.std(sizes)

    structure = np.ones((3,) * skeleton.ndim)
    stats['number of disjoint skeletons'] = ndi.label(skeleton, structure)[1]

    return stats
