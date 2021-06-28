from itertools import product
import functools
import operator
from dask import delayed
import dask.array as da
import dask.dataframe as dd
from dask_image.ndfilters import convolve
from dask_image.ndmeasure import label
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy import sparse

from .csr import Skeleton, csr_to_nbgraph, _build_skeleton_path_graph, _write_pixel_graph
from .nputil import raveled_steps_to_neighbors



def slices_from_chunks_overlap(chunks, array_shape, depth=1):
    """Translate chunks tuple to a set of slices in product order

    Parameters
    ----------
    chunks : tuple
        The chunks of the corresponding dask array.
    array_shape : tuple
        Shape of the corresponding dask array.
    depth : int
        The number of pixels to overlap, providing we're not at the array edge.

    Example
    -------
    >>> slices_from_chunks_overlap(((4,), (7, 7)), (4, 14), depth=1)  # doctest: +NORMALIZE_WHITESPACE
     [(slice(0, 5, None), slice(0, 8, None)),
      (slice(0, 5, None), slice(6, 15, None))]
    """
    cumdims = [da.slicing.cached_cumsum(bds, initial_zero=True) for bds in chunks]

    slices = []
    for starts, shapes in zip(cumdims, chunks):
        inner_slices = []
        for s, dim, maxshape in zip(starts, shapes, array_shape):
            slice_start = s
            slice_stop = s + dim
            if slice_start > 0:
                slice_start -= depth
            if slice_stop >= maxshape:
                slice_stop += depth
            inner_slices.append(slice(slice_start, slice_stop))
        slices.append(inner_slices)
    
    return list(product(*slices))


def graph_from_skelint(skelint):
    image = skelint

    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(functools.partial(operator.getitem, image),
            slices_from_chunks_overlap(image.chunks, image.shape, depth=1))
    )

    meta = dd.utils.make_meta([('row', np.int64), ('col', np.int64), ('data', np.float64)])  # it's very important to include meta
    intermediate_results = [dd.from_delayed(skeleton_graph_func(block), meta=meta) for _, block in block_iter]
    results = dd.concat(intermediate_results)
    results = results.drop_duplicates()
    # computes dask results, brings everything into memory before creating sparse graph
    k = len(results)
    row = np.array(results['row'])
    col = np.array(results['col'])
    data = np.array(results['data'])
    graph = sparse.coo_matrix((data[:k], (row[:k], col[:k]))).tocsr()
    return graph


@delayed
def skeleton_graph_func(skelint, spacing=1):
    ndim = skelint.ndim
    spacing = np.ones(ndim, dtype=float) * spacing
    num_edges = _num_edges(skelint.astype(bool))
    padded_skelint = np.pad(skelint, 1)  # pad image to prevent looparound errors
    steps, distances = raveled_steps_to_neighbors(padded_skelint.shape, ndim,
                                                  spacing=spacing)
    # from function skan.csr._pixel_graph
    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    data = np.empty(num_edges, dtype=float)
    k = _write_pixel_graph(padded_skelint, steps, distances, row, col, data)
    return pd.DataFrame({"row": row, "col": col, "data": data})


def _num_edges(skel):
    ndim = skel.ndim
    degree_kernel = np.ones((3,) * ndim)
    degree_kernel[(1,) * ndim] = 0  # remove centre pixel
    degree_image = ndi.convolve(skel.astype(int),
                                          degree_kernel,
                                          mode='constant') * skel
    num_edges = np.sum(degree_image)
    return int(num_edges)


class DaskSkeleton(Skeleton):
    def __init__(
        self,
        skel : da.Array,
        *,
        spacing=1,
        source_image=None,
        value_is_height=False,
        unique_junctions=True,
        ):
        self.skeleton_image = skel
        self.source_image = source_image  # if you have a raw data image
        if np.isscalar(spacing):
            self.spacing = [spacing] * skel.ndim  # pixel/voxel size
        else:
            self.spacing = spacing
        # skelint
        ndim = self.skeleton_image.ndim
        structure_kernel = np.zeros((3,) * ndim)
        structure_kernel[(1,) * ndim] = 1  # add centre pixel
        skelint, num_features = label(self.skeleton_image, structure=structure_kernel)
        self.skelint = skelint
        self.graph = graph_from_skelint(skelint)


        # degrees_image
        degree_kernel = np.ones((3,) * ndim)
        degree_kernel[(1,) * ndim] = 0  # remove centre pixel
        degrees_image = convolve(skel.astype(int), degree_kernel,
                                mode='constant') * skel
        self.degrees_image = degrees_image


        # Calculate the degrees attribute
        nonzero_degree_values = self.degrees_image[degrees_image > 0].compute()  # triggers Dask computation
        degrees = np.concatenate((np.array([0]), nonzero_degree_values))
        self.degrees = degrees

        # We also need to tell skan the non-zero pixel locations from our skeleton image.
        pixel_indices = np.nonzero(skel)
        pixel_coordinates = np.concatenate(([[0.] * ndim], np.transpose(pixel_indices)), axis=0) # triggers Dask computation 
        self.coordinates = pixel_coordinates

        if np.issubdtype(skel.dtype, np.floating):
            nonzero_pixel_intensity = skel.vindex[pixel_indices].compute()
            node_props = np.concatenate((np.array([0]), nonzero_pixel_intensity))  # add a dummy index
        else:
            node_props = None

        nbgraph = csr_to_nbgraph(self.graph, node_props=node_props)  # node_props=None is the default
        self.nbgraph = nbgraph

        # And last we can use some of skan's methods and functions directly 
        # to calculate the skeleton paths and branch distances.
        paths = _build_skeleton_path_graph(nbgraph, _buffer_size_offset=None)
        self.paths = paths
        self.n_paths = paths.shape[0]

        # MUST reset both distances_intialized AND the empty numpy array to calculate the branch length
        self._distances_initialized = False
        self.distances = np.empty(self.n_paths, dtype=float)

