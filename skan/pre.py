import numpy as np
from scipy import spatial, ndimage as ndi
from skimage import filters, restoration

# Temporary until skimage 0.13 is out
from .vendored.thresholding import threshold_sauvola, threshold_niblack


SMOOTH_METHODS = {
    'Gaussian': filters.gaussian,
    'TV': restoration.denoise_tv_bregman,
}


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


def threshold(image, *, sigma=0., radius=0, offset=0.,
              method='sauvola', smooth_method='Gaussian'):
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
        result in fewer pixels above the threshold.
    method: {'sauvola', 'niblack', 'median'}
        Which method to use for thresholding. Sauvola is 100x faster, but
        median might be more accurate.
    smooth_method: {'Gaussian', 'TV'}
        Which method to use for smoothing. Choose from Gaussian smoothing
        and total variation denoising.

    Returns
    -------
    thresholded : image of bool, same shape as `image`
        The thresholded image.

    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/user_guide/data_types.html
    """
    if sigma > 0:
        if smooth_method.lower() == 'gaussian':
            image = filters.gaussian(image, sigma=sigma)
        elif smooth_method.lower() == 'tv':
            image = restoration.denoise_tv_bregman(image, weight=sigma)
    if radius == 0:
        t = filters.threshold_otsu(image) + offset
    else:
        if method == 'median':
            footprint = hyperball(image.ndim, radius=radius)
            t = ndi.median_filter(image, footprint=footprint) + offset
        elif method == 'sauvola':
            w = 2 * radius + 1
            t = threshold_sauvola(image, window_size=w, k=offset)
        elif method == 'niblack':
            w = 2 * radius + 1
            t = threshold_niblack(image, window_size=w, k=offset)
        else:
            raise ValueError('Unknown method %s. Valid methods are median,'
                             'niblack, and sauvola.' % method)
    thresholded = image > t
    return thresholded
