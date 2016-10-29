import collections
import itertools
import numpy as np
import pandas as pd
from scipy import sparse, ndimage as ndi
import networkx as nx
import numba


## CSGraph and Numba-based implementation


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


@numba.jit(nopython=True, cache=True, nogil=True)
def write_pixel_graph(image, indices, steps, distances, row, col, data):
    image = image.ravel()
    n_neighbors = steps.size
    n_nodes = indices.size
    k = 0
    for h in range(n_nodes):
        i = image[h]
        if image[i] != 0:
            for j in range(n_neighbors):
                n = steps[j] + i
                if image[n] != 0:
                    row[k] = image[i]
                    col[k] = image[n]
                    data[k] = distances[j]
                    k += 1


def skeleton_to_csgraph(skel):
    skel = skel.astype(bool)  # ensure we have a bool image
                              # since we later use it for bool indexing
    ndim = skel.ndim
    skelint = np.zeros(skel.shape, int)
    skelint[skel] = np.arange(1, np.sum(skel) + 1)
    skelint = pad(skelint, 0)

    degree_kernel = np.ones((3,) * ndim)
    degree_kernel.ravel()[3**ndim // 2] = 0  # remove centre pix
    degree_image = ndi.convolve(skel.astype(int), degree_kernel,
                                mode='constant')
    num_edges = np.sum(degree_image)  # *2, which is how many we need to store
    row, col = np.zeros(num_edges, dtype=int), np.zeros(num_edges, dtype=int)
    data = np.zeros(num_edges, dtype=float)
    steps, distances = raveled_steps_to_neighbors(skelint.shape, ndim)
    write_pixel_graph(skelint, np.flatnonzero(skelint),
                      steps, distances, row, col, data)
    return sparse.coo_matrix((data, (row, col))).tocsr()


## NetworkX-based implementation


def _add_skeleton_edges(values, graph, distances):
    values = values.astype(int)
    center_idx = len(values) // 2
    center = values[center_idx]
    if center == 0:
        return 0
    count = 0.
    for value, distance in zip(values, distances):
        if value != center and value != 0:
            graph.add_edge(center, value, weight=distance)
            count += 1
    node = graph.node[center]
    if count <= 1:
        node['type'] = 'tip'
    elif count == 2:
        node['type'] = 'path'
    else:
        node['type'] = 'junction'
    return count


def _neighbor_distances(ndim):
    center = np.ones((3,) * ndim, dtype=bool)
    center.ravel()[center.size//2] = False
    out = ndi.distance_transform_edt(center).ravel()
    return out


def skeleton_to_nx(skel):  # to do: add pixel spacing
    g = nx.Graph()
    distances = _neighbor_distances(skel.ndim)
    skelint = np.zeros(skel.shape, int)
    num_nodes = np.sum(skel)
    g.add_nodes_from(range(1, num_nodes + 1))
    skelint[skel] = np.arange(1, num_nodes + 1)
    counts = ndi.generic_filter(skelint, function=_add_skeleton_edges,
                                size=3, mode='constant', cval=0,
                                extra_arguments=(g, distances))
    return g, counts, skelint


def branch_statistics(g):
    """Compute the length and type of each branch in a skeleton graph.

    Parameters
    ----------
    g : nx.Graph
        A skeleton graph. Its nodes must be marked as 'tip', 'path', or
        'junction'.

    Returns
    -------
    branches : array of float, shape (N, 4, 2)
        An array containing branch endpoint IDs, length, and branch type.
        The types are:
        - tip-tip (0)
        - tip-junction (1)
        - junction-junction (2)
    """
    visited = np.zeros(max(g) + 1, dtype=bool)
    type_dict = {'tiptip': 0, 'tipjunction': 1, 'junctiontip': 1,
                 'junctionjunction': 2, 'pathpath': 3}
    result = []
    for node, data in g.nodes_iter(data=True):
        if data['type'] == 'path' and not visited[node]:
            # we expand the path in either direction
            visited[node] = True
            left, right = g.neighbors(node)
            id0, d0, kind0 = _expand_path(g, node, left, visited)
            id1, d1, kind1 = _expand_path(g, node, right, visited)
            result.append([id0, id1, d0 + d1, type_dict[kind0 + kind1]])
    return np.array(result)


def _expand_path(g, source, step, visited):
    d = g[source][step]['weight']
    while g.node[step]['type'] == 'path' and not visited[step]:
        n1, n2 = g.neighbors(step)
        nextstep = n1 if n1 != source else n2
        source, step = step, nextstep
        d += g[source][step]['weight']
        visited[source] = True
    return source, d, g.node[step]['type']


def summarise(skelimage):
    ndim = skelimage.ndim
    g, counts, skelimage_labeled = skeleton_to_nx(skelimage)
    coords = np.nonzero(skelimage)
    ids = skelimage_labeled[coords]
    sorted_coords = np.transpose(coords)[np.argsort(ids)]
    tables = []
    for i, cc in enumerate(nx.connected_component_subgraphs(g)):
        stats = branch_statistics(cc)
        if stats.size == 0:
            continue
        coords0 = sorted_coords[stats[:, 0].astype(int) - 1]
        coords1 = sorted_coords[stats[:, 1].astype(int) - 1]
        distances = np.sqrt(np.sum((coords0 - coords1)**2, axis=1))
        skeleton_id = np.full(distances.shape, i, dtype=float)
        tables.append(np.column_stack((skeleton_id, stats,
                                       coords0, coords1, distances)))
    columns = (['skeleton-id', 'node-id-0', 'node-id-1', 'branch-distance',
                'branch-type'] +
               ['coord-0-%i' % i for i in range(ndim)] +
               ['coord-1-%i' % i for i in range(ndim)] +
               ['euclidean-distance'])
    column_types = [int, int, int, float, int] + 2*ndim*[int] + [float]
    arr = np.row_stack(tables).T
    data_dict = {col: dat.astype(dtype)
                 for col, dat, dtype in zip(columns, arr, column_types)}
    df = pd.DataFrame(data_dict)
    return df
