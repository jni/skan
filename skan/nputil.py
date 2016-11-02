import collections
import itertools
import numpy as np

# adapted from github.com/janelia-flyem/gala
def smallest_int_dtype(number, signed=False, min_dtype=np.int8):
    if number < 0:
        signed = True
    if not signed:
        if number <= np.iinfo(np.uint8).max:
            dtype = np.uint8
        if number <= np.iinfo(np.uint16).max:
            dtype = np.uint16
        if number <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        if number <= np.iinfo(np.uint64).max:
            dtype = np.uint64
    else:
        if np.iinfo(np.int8).min <= number <= np.iinfo(np.int8).max:
            dtype = np.int8
        if np.iinfo(np.int16).min <= number <= np.iinfo(np.int16).max:
            dtype = np.int16
        if np.iinfo(np.int32).min <= number <= np.iinfo(np.int32).max:
            dtype = np.int32
        if np.iinfo(np.int64).min <= number <= np.iinfo(np.int64).max:
            dtype = np.int64
    if np.iinfo(dtype).max < np.iinfo(min_dtype).max:
        dtype = min_dtype
    return dtype


# adapted from github.com/janelia-flyem/gala
def pad(ar, vals, axes=None):
    if ar.size == 0:
        return ar
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


def raveled_steps_to_neighbors(shape, connectivity=1, order='C',
                               return_distances=True):
    if order == 'C':
        dims = shape[-1:0:-1]
    else:
        dims = shape[:-1]
    stepsizes = np.cumprod((1,) + dims)[::-1]
    steps = [stepsizes, -stepsizes]
    distances = [1] * 2 * stepsizes.size
    for nhops in range(2, connectivity + 1):
        prod = np.array(list(itertools.product(*[[1, -1]] * nhops)))
        multisteps = np.array(list(itertools.combinations(stepsizes, nhops))).T
        steps.append((prod @ multisteps).ravel())
        distances.extend([np.sqrt(nhops)] * steps[-1].size)
    if return_distances:
        return np.concatenate(steps).astype(int), np.array(distances)
    else:
        return np.concatenate(steps).astype(int)

