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
                 'junctionjunction': 2}
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
    while g.node[step]['type'] == 'path':
        n1, n2 = g.neighbors(step)
        nextstep = n1 if n1 != source else n2
        source, step = step, nextstep
        d += g[source][step]['weight']
        visited[source] = True
    return source, d, g.node[step]['type']


def summarise(skelimage):
    g, counts, skelimage_labeled = skeleton_to_nx(skelimage)
    coords = np.nonzero(skelimage)
    ids = skelimage_labeled[coords]
    sorted_coords = np.transpose(coords)[np.argsort(ids)]
    tables = []
    for i, cc in enumerate(nx.connected_component_subgraphs(g)):
        stats = branch_statistics(cc)
        if stats.size == 0:
            continue
        coords0 = sorted_coords[stats[:, 0].astype(int)]
        coords1 = sorted_coords[stats[:, 1].astype(int)]
        distances = np.sqrt(np.sum((coords0 - coords1)**2, axis=1))
        skeleton_id = np.full(distances.shape, i, dtype=float)
        tables.append(np.column_stack((skeleton_id, stats,
                                       coords0, coords1, distances)))
    return np.row_stack(tables)
