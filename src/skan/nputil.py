import itertools
import numpy as np
from scipy import ndimage as ndi


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


def raveled_steps_to_neighbors(
        shape, connectivity=1, *, order='C', spacing=1, return_distances=True
        ):
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
        dhopsq = np.array(list(itertools.combinations(spacing**2, nhops))).T
        steps.append((prod @ multisteps).ravel())
        distances.append(np.sqrt(np.abs(prod) @ dhopsq).ravel())
    if return_distances:
        return (np.concatenate(steps).astype(int), np.concatenate(distances))
    else:
        return np.concatenate(steps).astype(int)


def _validate_connectivity(image_dim, connectivity, offset):
    """Convert any valid connectivity to a footprint and offset.

    Parameters
    ----------
    image_dim : int
        The number of dimensions of the input image.
    connectivity : int, array, or None
        The neighborhood connectivity. An integer is interpreted as in
        ``scipy.ndimage.generate_binary_structure``, as the maximum number
        of orthogonal steps to reach a neighbor. An array is directly
        interpreted as a footprint and its shape is validated against
        the input image shape. ``None`` is interpreted as a connectivity of 1.
    offset : tuple of int, or None
        The coordinates of the center of the footprint.

    Returns
    -------
    c_connectivity : array of bool
        The footprint (structuring element) corresponding to the input
        `connectivity`.
    offset : array of int
        The offset corresponding to the center of the footprint.

    Raises
    ------
    ValueError:
        If the image dimension and the connectivity or offset dimensions don't
        match.
    """
    if connectivity is None:
        connectivity = 1

    if np.isscalar(connectivity):
        c_connectivity = ndi.generate_binary_structure(image_dim, connectivity)
    else:
        c_connectivity = np.array(connectivity, bool)
        if c_connectivity.ndim != image_dim:
            raise ValueError("Connectivity dimension must be same as image")

    if offset is None:
        if any([x % 2 == 0 for x in c_connectivity.shape]):
            raise ValueError(
                    "Connectivity array must have an unambiguous "
                    "center"
                    )

        offset = np.array(c_connectivity.shape) // 2

    return c_connectivity, offset


def _raveled_offsets_and_distances(
        image_shape,
        *,
        footprint=None,
        connectivity=1,
        center=None,
        spacing=None,
        order='C',
        ):
    """Compute offsets to neighboring pixels in raveled coordinate space.

    This function also returns the corresponding distances from the center
    pixel given a spacing (assumed to be 1 along each axis by default).

    Parameters
    ----------
    image_shape : tuple of int
        The shape of the image for which the offsets are being computed.
    footprint : array of bool
        The footprint of the neighborhood, expressed as an n-dimensional array
        of 1s and 0s. If provided, the connectivity argument is ignored.
    connectivity : {1, ..., ndim}
        The square connectivity of the neighborhood: the number of orthogonal
        steps allowed to consider a pixel a neighbor. See
        `scipy.ndimage.generate_binary_structure`. Ignored if footprint is
        provided.
    center : tuple of int
        Tuple of indices to the center of the footprint. If not provided, it
        is assumed to be the center of the footprint, either provided or
        generated by the connectivity argument.
    spacing : tuple of float
        The spacing between pixels/voxels along each axis.
    order : 'C' or 'F'
        The ordering of the array, either C or Fortran ordering.

    Returns
    -------
    raveled_offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their distance from the center.
    distances : ndarray
        The pixel distances correspoding to each offset.

    Notes
    -----
    This function will return values even if `image_shape` contains a dimension
    length that is smaller than `footprint`.

    Examples
    --------
    >>> off, d = _raveled_offsets_and_distances(
    ...         (4, 5), footprint=np.ones((4, 3)), center=(1, 1)
    ...         )
    >>> off  # doctest: +SKIP
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    >>> d[0]
    1.0
    >>> d[-1]  # distance from (1, 1) to (3, 2)
    2.236...
    """
    ndim = len(image_shape)
    if footprint is None:
        footprint = ndi.generate_binary_structure(
                rank=ndim, connectivity=connectivity
                )
    if center is None:
        center = np.array(footprint.shape) // 2
    if not footprint.ndim == ndim == len(center):
        raise ValueError(
                "number of dimensions in image shape, footprint and its"
                "center index does not match"
                )

    footprint_indices = np.stack(np.nonzero(footprint), axis=-1)
    offsets = footprint_indices - center

    if order == 'F':
        offsets = offsets[:, ::-1]
        image_shape = image_shape[::-1]
    elif order != 'C':
        raise ValueError("order must be 'C' or 'F'")

    # Scale offsets in each dimension and sum
    ravel_factors = image_shape[1:] + (1,)
    ravel_factors = np.cumprod(ravel_factors[::-1])[::-1]
    raveled_offsets = (offsets * ravel_factors).sum(axis=1)

    # Sort by distance
    if spacing is None:
        spacing = np.ones(ndim)
    weighted_offsets = offsets * spacing
    distances = np.sqrt(np.sum(weighted_offsets**2, axis=1))
    sorted_raveled_offsets = raveled_offsets[np.argsort(distances)]
    sorted_distances = np.sort(distances)

    # If any dimension in image_shape is smaller than footprint.shape
    # duplicates might occur, remove them
    if any(x < y for x, y in zip(image_shape, footprint.shape)):
        # np.unique reorders, which we don't want
        _, indices = np.unique(sorted_raveled_offsets, return_index=True)
        sorted_raveled_offsets = sorted_raveled_offsets[np.sort(indices)]
        sorted_distances = sorted_distances[np.sort(indices)]

    # Remove "offset to center"
    sorted_raveled_offsets = sorted_raveled_offsets[1:]
    sorted_distances = sorted_distances[1:]

    return sorted_raveled_offsets, sorted_distances
