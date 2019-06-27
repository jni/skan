import itertools
import numpy as np
from skimage.transform import integral_image
from skimage.util import dtype_limits
import numba


def broadcast_mgrid(arrays):
    shape = tuple(map(len, arrays))
    ndim = len(shape)
    result = []
    for i, arr in enumerate(arrays, start=1):
        reshaped = np.broadcast_to(arr[(...,) + (np.newaxis,) * (ndim - i)],
                                   shape)
        result.append(reshaped)
    return result


@numba.jit(nopython=True, cache=True, nogil=True)
def _correlate_sparse_offsets(input, indices, offsets, values, output):
    for off, val in zip(offsets, values):
        # this loop order optimises cache access, gives 10x speedup
        for i, j in enumerate(indices):
            output[i] += input[j + off] * val


def correlate_sparse(image, kernel, mode='reflect'):
    """Compute valid cross-correlation of `padded_array` and `kernel`.

    This function is *fast* when `kernel` is large with many zeros.

    See ``scipy.ndimage.correlate`` for a description of cross-correlation.

    Parameters
    ----------
    image : array of float, shape (M, N,[ ...,] P)
        The input array. It should be already padded, as a margin of the
        same shape as kernel (-1) will be stripped off.
    kernel : array of float, shape (Q, R,[ ...,] S)
        The kernel to be correlated. Must have the same number of
        dimensions as `padded_array`. For high performance, it should
        be sparse (few nonzero entries).
    mode : string, optional
        See `np.pad` for valid modes. Additionally, mode 'valid' is
        accepted, in which case no padding is applied and the result is
        the result for the smaller image for which the kernel is entirely
        inside the original data.

    Returns
    -------
    result : array of float, shape (M, N,[ ...,] P)
        The result of cross-correlating `image` with `kernel`. If mode
        'valid' is used, the resulting shape is (M-Q+1, N-R+1,[ ...,] P-S+1).
    """
    if mode == 'valid':
        padded_image = image
    else:
        w = kernel.shape[0] // 2
        padded_image = np.pad(image, (w, w-1), mode=mode)
    indices = np.nonzero(kernel)
    offsets = np.ravel_multi_index(indices, padded_image.shape)
    values = kernel[indices]
    result = np.zeros([a - b + 1
                       for a, b in zip(padded_image.shape, kernel.shape)])
    corner_multi_indices = broadcast_mgrid([np.arange(i)
                                            for i in result.shape])
    corner_indices = np.ravel_multi_index(corner_multi_indices,
                                          padded_image.shape).ravel()
    _correlate_sparse_offsets(padded_image.ravel(), corner_indices,
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

    sum_full = correlate_sparse(integral, kern, mode='valid')
    m = sum_full / (w ** image.ndim)
    sum_sq_full = correlate_sparse(integral_sq, kern, mode='valid')
    g2 = sum_sq_full / (w ** image.ndim)
    s = np.sqrt(g2 - m * m)
    return m, s


def threshold_niblack(image, window_size=15, k=0.2):
    """Apply Niblack local threshold to an array. [1]_

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
    """Apply Sauvola local threshold to an array. [2]_

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
    .. [2] J. Sauvola and M. Pietikainen, "Adaptive document image
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
