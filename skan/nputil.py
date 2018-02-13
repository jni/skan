import collections
import itertools
import numpy as np

# adapted from github.com/janelia-flyem/gala
def smallest_int_dtype(number, *, signed=False, min_dtype=np.int8):
    """Return the smallest numpy integer dtype that can represent `number`.

    Parameters
    ----------
    number : int
        The minimum number to be represented by dtype.
    signed : bool, optional
        Whether a signed dtype is required.
    min_dtype : numpy dtype, optional
        Specify a minimum dtype in case `number` is not the absolute
        maximum that the user wants to represent.

    Returns
    -------
    dtype : numpy dtype
        The required data type.

    Examples
    --------
    >>> smallest_int_dtype(8)
    <class 'numpy.uint8'>
    >>> smallest_int_dtype(2**9)
    <class 'numpy.uint16'>
    >>> smallest_int_dtype(2**17)
    <class 'numpy.uint32'>
    >>> smallest_int_dtype(2**33)
    <class 'numpy.uint64'>
    >>> smallest_int_dtype(8, signed=True)
    <class 'numpy.int8'>
    >>> smallest_int_dtype(8, signed=True, min_dtype=np.int16)
    <class 'numpy.int16'>
    >>> smallest_int_dtype(-2**9)
    <class 'numpy.int16'>
    >>> smallest_int_dtype(-2**17)
    <class 'numpy.int32'>
    >>> smallest_int_dtype(-2**33)
    <class 'numpy.int64'>
    """
    if number < 0:
        signed = True
        number = abs(number)
    if not signed:
        if number <= np.iinfo(np.uint8).max:
            dtype = np.uint8
        elif number <= np.iinfo(np.uint16).max:
            dtype = np.uint16
        elif number <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        else:  # number <= np.iinfo(np.uint64).max:
            dtype = np.uint64
    else:
        if np.iinfo(np.int8).min <= number <= np.iinfo(np.int8).max:
            dtype = np.int8
        elif np.iinfo(np.int16).min <= number <= np.iinfo(np.int16).max:
            dtype = np.int16
        elif np.iinfo(np.int32).min <= number <= np.iinfo(np.int32).max:
            dtype = np.int32
        else:  # if np.iinfo(np.int64).min <= number <= np.iinfo(np.int64).max:
            dtype = np.int64
    if np.iinfo(dtype).max < np.iinfo(min_dtype).max:
        dtype = min_dtype
    return dtype


# adapted from github.com/janelia-flyem/gala
def pad(ar, vals, *, axes=None):
    """Pad an array with values in `vals` along `axes`.

    Parameters
    ----------
    ar : array, shape (M, N, ...)
        The input array.
    vals : int or iterable of int, shape (K,)
        The values to pad with.
    axes : int in {0, ..., `ar.ndim`}, or iterable thereof, optional
        The axes of `ar` to pad. If None, pad along all axes.

    Returns
    -------
    ar2 : array, shape (M+2K, N+2K, ...)
        The padded array.

    Examples
    --------
    >>> ar = np.array([4, 5, 6])
    >>> pad(ar, 0)
    array([0, 4, 5, 6, 0])
    >>> pad(ar, [0, 1])
    array([1, 0, 4, 5, 6, 0, 1])
    >>> ar = np.array([[4, 5, 6]])
    >>> pad(ar, 0)
    array([[0, 0, 0, 0, 0],
           [0, 4, 5, 6, 0],
           [0, 0, 0, 0, 0]])
    >>> pad(ar, 0, axes=1)
    array([[0, 4, 5, 6, 0]])
    """
    if axes is None:
        axes = list(range(ar.ndim))
    if not isinstance(vals, collections.Iterable):
        vals = [vals]
    if not isinstance(axes, collections.Iterable):
        axes = [axes]
    p = len(vals)
    newshape = np.array(ar.shape)
    for ax in axes:
        newshape[ax] += 2*p
    vals = np.reshape(vals, (p,) + (1,) * (ar.ndim-1))
    new_dtype = ar.dtype
    if np.issubdtype(new_dtype, np.integer):
        maxval = max([np.max(vals), np.max(ar)])
        minval = min([np.min(vals), np.min(ar)])
        signed = (minval < 0)
        maxval = max(abs(minval), maxval)
        new_dtype = smallest_int_dtype(maxval, signed=signed,
                                       min_dtype=new_dtype)
    ar2 = np.empty(newshape, dtype=new_dtype)
    center = np.ones(newshape, dtype=bool)
    for ax in axes:
        ar2.swapaxes(0, ax)[p-1::-1,...] = vals
        ar2.swapaxes(0, ax)[-p:,...] = vals
        center.swapaxes(0, ax)[p-1::-1,...] = False
        center.swapaxes(0, ax)[-p:,...] = False
    ar2[center] = ar.ravel()
    return ar2


def raveled_steps_to_neighbors(shape, connectivity=1, *, order='C', spacing=1,
                               return_distances=True):
    """Return raveled coordinate steps for given array shape and neighborhood.

    Parameters
    ----------
    shape : tuple of int
        The array shape.
    connectivity : {1, ..., len(shape)}, optional
        The n-dimensional connectivity. See
        `scipy.ndimage.generate_binary_structure` for more.
    order : {'C', 'F'}, optional
        The ordering of the array, either C or Fortran.
    spacing : float, or array-like of float, shape `len(shape)`
        The spacing of the pixels along each dimension.
    return_distances : bool, optional
        If True (default), return also the Euclidean distance to each
        neighbor.

    Returns
    -------
    steps : array of int, shape (K,)
        Each value in `steps` moves from a central pixel to a
        `connectivity`-neighbor in an array of shape `shape`.
    distances : array of float, shape (K,), optional
        The Euclidean distance corresponding to each step. This is only
        returned if `return_distances` is True.

    Examples
    --------
    >>> raveled_steps_to_neighbors((5,), 1)
    (array([ 1, -1]), array([1., 1.]))
    >>> raveled_steps_to_neighbors((2, 3), 2, return_distances=False)
    array([ 3,  1, -3, -1,  4,  2, -2, -4])
    >>> raveled_steps_to_neighbors((2, 3), 1, order='F')[0]
    array([ 2,  1, -2, -1])

    Using `spacing` we can obtain different distance values along different
    axes:

    >>> raveled_steps_to_neighbors((3, 4, 5), spacing=[5, 1, 1])
    (array([ 20,   5,   1, -20,  -5,  -1]), array([5., 1., 1., 5., 1., 1.]))
    """
    spacing = np.ones(len(shape), dtype=float) * spacing
    if order == 'C':
        dims = shape[-1:0:-1]
    else:
        dims = shape[:-1]
    stepsizes = np.cumprod((1,) + dims)[::-1]
    steps = [stepsizes, -stepsizes]
    distances = [spacing, spacing]
    for nhops in range(2, connectivity + 1):
        prod = np.array(list(itertools.product(*[[1, -1]] * nhops)))
        multisteps = np.array(list(itertools.combinations(stepsizes, nhops))).T
        dhopsq = np.array(list(itertools.combinations(spacing ** 2, nhops))).T
        steps.append((prod @ multisteps).ravel())
        distances.append(np.sqrt(np.abs(prod) @ dhopsq).ravel())
    if return_distances:
        return (np.concatenate(steps).astype(int),
                np.concatenate(distances))
    else:
        return np.concatenate(steps).astype(int)

