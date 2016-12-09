import numpy as np
from scipy import spatial, ndimage as ndi
from skimage import filters, img_as_ubyte


def hyperball(ndim, radius):
    """Return a binary morphological filter containing pixels within `radius`.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the filter.
    radius : int
        The radius of the filter.

    Returns
    -------
    ball : array of bool, shape [2 * radius + 1,] * ndim
        The required structural element
    """
    size = 2 * radius + 1
    center = [(radius,) * ndim]

    coords = np.mgrid[[slice(None, size),] * ndim].reshape(ndim, -1).T
    distances = np.ravel(spatial.distance_matrix(coords, center))
    selector = distances <= radius

    ball = np.zeros((size,) * ndim, dtype=bool)
    ball.ravel()[selector] = True
    return ball



def threshold(image, *, sigma=0., radius=0, offset=0.):
    """Use scikit-image filters to "intelligently" threshold an image.

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

    Returns
    -------
    thresholded : image of bool, same shape as `image`
        The thresholded image.

    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/user_guide/data_types.html
    """
    if sigma > 0:
        image = filters.gaussian(image, sigma=sigma)
    image = img_as_ubyte(image)
    if len(np.unique(image)) == 1:
        return np.zeros(image.shape, dtype=bool)
    if radius > 0:
        footprint = hyperball(image.ndim, radius=radius)
        t = ndi.median_filter(image, footprint=footprint) + offset
    else:
        t = filters.threshold_otsu(image) + offset
    thresholded = image > t
    return thresholded
