import itertools
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import integral_image
from skimage import util
from skimage.util import dtype_limits
import numba


@numba.jit(nopython=True, cache=True, nogil=True)
def _correlate_nonzeros_offset(input, indices, offsets, values, output):
    for i, j in enumerate(indices):
        for off, val in zip(offsets, values):
            output[i] += input[j + off] * val


def correlate_nonzeros(padded_array, kernel):
    indices = np.nonzero(kernel)
    offsets = np.ravel_multi_index(indices, padded_array.shape)
    values = kernel[indices]
    result = np.zeros(np.array(padded_array.shape) - np.array(kernel.shape)
                      + 1)
    corner_multi_indices = np.mgrid[[slice(None, i) for i in result.shape]]
    corner_indices = np.ravel_multi_index(corner_multi_indices,
                                          padded_array.shape).ravel()
    _correlate_nonzeros_offset(padded_array.ravel(), corner_indices,
                               offsets, values, result.ravel())
    return result


def _mean_std(image, w):
    """Return local mean and standard deviation of each pixel using a
    neighborhood defined by a rectangular window with size w times w.
    The algorithm uses integral images to speedup computation. This is
    used by threshold_niblack and threshold_sauvola.

    Parameters
    ----------
    image : ndarray
        Input image.
    w : int
        Odd window size (e.g. 3, 5, 7, ..., 21, ...).

    Returns
    -------
    m : 2-D array of same size of image with local mean values.
    s : 2-D array of same size of image with local standard
        deviation values.

    References
    ----------
    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient
           implementation of local adaptive thresholding techniques
           using integral images." in Document Recognition and
           Retrieval XV, (San Jose, USA), Jan. 2008.
           DOI:10.1117/12.767755
    """
    if w == 1 or w % 2 == 0:
        raise ValueError(
            "Window size w = %s must be odd and greater than 1." % w)

    left_pad = w // 2 + 1
    right_pad = w // 2
    padded = np.pad(image.astype('float'), (left_pad, right_pad),
                    mode='reflect')
    padded_sq = padded * padded

    integral = integral_image(padded)
    integral_sq = integral_image(padded_sq)

    kern = np.zeros((w + 1,) * image.ndim)
    for indices in itertools.product(*([[0, -1]] * image.ndim)):
        kern[indices] = (-1) ** (image.ndim % 2 != np.sum(indices) % 2)

    sum_full = correlate_nonzeros(integral, kern)
    m = sum_full / (w ** image.ndim)
    sum_sq_full = correlate_nonzeros(integral_sq, kern)
    g2 = sum_sq_full / (w ** image.ndim)
    s = np.sqrt(g2 - m * m)
    return m, s


def threshold_niblack(image, window_size=15, k=0.2):
    """Applies Niblack local threshold to an array.

    A threshold T is calculated for every pixel in the image using the
    following formula:

    T = m(x,y) - k * s(x,y)

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.

    Parameters
    ----------
    image: (N, M) ndarray
        Grayscale input image.
    window_size : int, optional
        Odd size of pixel neighborhood window (e.g. 3, 5, 7...).
    k : float, optional
        Value of parameter k in threshold formula.

    Returns
    -------
    threshold : (N, M) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    References
    ----------
    .. [1] Niblack, W (1986), An introduction to Digital Image
           Processing, Prentice-Hall.

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> binary_image = threshold_niblack(image, window_size=7, k=0.1)
    """
    m, s = _mean_std(image, window_size)
    return m - k * s


def threshold_sauvola(image, window_size=15, k=0.2, r=None):
    """Applies Sauvola local threshold to an array. Sauvola is a
    modification of Niblack technique.

    In the original method a threshold T is calculated for every pixel
    in the image using the following formula:

    T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.
    R is the maximum standard deviation of a greyscale image.

    Parameters
    ----------
    image: (N, M) ndarray
        Grayscale input image.
    window_size : int, optional
        Odd size of pixel neighborhood window (e.g. 3, 5, 7...).
    k : float, optional
        Value of the positive parameter k.
    r : float, optional
        Value of R, the dynamic range of standard deviation.
        If None, set to the half of the image dtype range.
    offset : float, optional
        Constant subtracted from obtained local thresholds.

    Returns
    -------
    threshold : (N, M) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    References
    ----------
    .. [1] J. Sauvola and M. Pietikainen, "Adaptive document image
           binarization," Pattern Recognition 33(2),
           pp. 225-236, 2000.
           DOI:10.1016/S0031-3203(99)00055-2

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> binary_sauvola = threshold_sauvola(image,
    ...                                    window_size=15, k=0.2)
    """
    if r is None:
        imin, imax = dtype_limits(image, clip_negative=False)
        r = 0.5 * (imax - imin)
    m, s = _mean_std(image, window_size)
    return m * (1 + k * ((s / r) - 1))
