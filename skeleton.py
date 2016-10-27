import numpy as np
from scipy import sparse, ndimage as ndi
import networkx as nx
import numba


@numba.jit
def get_nz_neighbors(arr, idx, steps, out):
    pass


def skeleton_to_csgraph(skel):
    skelint = np.zeros(skel.shape, int)
    skelint[skel] = np.arange(1, np.sum(skel) + 1)


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
    ndi.generic_filter(skelint, function=_add_skeleton_edges,
                       size=3, mode='constant', cval=0,
                       extra_arguments=(g, distances))
    return g
