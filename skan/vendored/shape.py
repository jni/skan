import numpy as np
from scipy import ndimage as ndi
from itertools import combinations_with_replacement
from skimage import img_as_float
from skimage.feature import hessian_matrix_eigvals


def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc'):
    """Compute Hessian matrix.

    The Hessian matrix is defined as::

        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective x- and y-directions.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'xy', 'rc'}, optional
        This parameter allows for the use of reverse or forward order of
        the image axes in gradient computation. 'xy' indicates the usage
        of the last axis initially (Hxx, Hxy, Hyy), whilst 'rc' indicates
        the use of the first axis initially (Hrr, Hrc, Hcc).

    Returns
    -------
    Hrr : ndarray
        Element of the Hessian matrix for each pixel in the input image.
    Hrc : ndarray
        Element of the Hessian matrix for each pixel in the input image.
    Hcc : ndarray
        Element of the Hessian matrix for each pixel in the input image.

    Examples
    --------
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1)
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    """

    image = img_as_float(image)

    gaussian_filtered = ndi.gaussian_filter(image, sigma=sigma,
                                            mode=mode, cval=cval)

    gradients = np.gradient(gaussian_filtered)
    axes = range(image.ndim)

    if order == 'rc':
        axes = reversed(axes)

    H_elems = [np.gradient(gradients[ax0], axis=ax1)
               for ax0, ax1 in combinations_with_replacement(axes, 2)]

    return H_elems


def shape_index(image, sigma=1, mode='constant', cval=0):
    """Compute the shape index.

    The shape index, as defined by Koenderink & van Doorn [1]_, is a
    single valued measure of local curvature, assuming the image as a 3D plane
    with intensities representing heights.

    It is derived from the eigen values of the Hessian, and its
    value ranges from -1 to 1 (and is undefined (=NaN) in *flat* regions),
    with following ranges representing following shapes:

    .. table:: Ranges of the shape index and corresponding shapes.

      ===================  =============
      Interval (s in ...)  Shape
      ===================  =============
      [  -1, -7/8)         Spherical cup
      [-7/8, -5/8)         Through
      [-5/8, -3/8)         Rut
      [-3/8, -1/8)         Saddle rut
      [-1/8, +1/8)         Saddle
      [+1/8, +3/8)         Saddle ridge
      [+3/8, +5/8)         Ridge
      [+5/8, +7/8)         Dome
      [+7/8,   +1]         Spherical cap
      ===================  =============

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used for
        smoothing the input data before Hessian eigen value calculation.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    s : ndarray
        Shape index

    References
    ----------
    .. [1] Koenderink, J. J. & van Doorn, A. J.,
           "Surface shape and curvature scales",
           Image and Vision Computing, 1992, 10, 557-564.
           DOI:10.1016/0262-8856(92)90076-F

    Examples
    --------
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> s = shape_index(square, sigma=0.1)
    >>> s
    array([[ nan,  nan, -0.5,  nan,  nan],
           [ nan, -0. ,  nan, -0. ,  nan],
           [-0.5,  nan, -1. ,  nan, -0.5],
           [ nan, -0. ,  nan, -0. ,  nan],
           [ nan,  nan, -0.5,  nan,  nan]])
    """
    R = hessian_matrix(image, sigma=sigma, mode=mode, cval=cval,
                                   order='rc')
    l1, l2 = hessian_matrix_eigvals(*R)

    return (2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1))
