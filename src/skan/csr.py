import numpy as np
import pandas as pd
from scipy import sparse, ndimage as ndi
from scipy.sparse import csgraph
from skimage.graph import pixel_graph
import numba

from .summary_utils import find_main_branches


csr_spec_float = [
        ('indptr', numba.int32[:]),
        ('indices', numba.int32[:]),
        ('data', numba.float64[:]),
        ('shape', numba.int32[:]),
        ('node_properties', numba.float64[:]),
        ]  # yapf: disable

csr_spec_bool = [
        ('indptr', numba.int32[:]),
        ('indices', numba.int32[:]),
        ('data', numba.bool_[:]),
        ('shape', numba.int32[:]),
        ('node_properties', numba.float64[:]),
        ]  # yapf: disable


class NBGraphBase:
    def __init__(self, indptr, indices, data, shape, node_props):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape
        self.node_properties = node_props

    def edge(self, i, j):
        return _csrget(self.indices, self.indptr, self.data, i, j)

    def set_edge(self, i, j, value):
        return _csrset(self.indices, self.indptr, self.data, i, j, value)

    def neighbors(self, row):
        loc, stop = self.indptr[row], self.indptr[row + 1]
        return self.indices[loc:stop]

    @property
    def has_node_props(self):
        return self.node_properties.strides != (0,)


NBGraph = numba.experimental.jitclass(NBGraphBase, csr_spec_float)
NBGraphBool = numba.experimental.jitclass(NBGraphBase, csr_spec_bool)


def csr_to_nbgraph(csr, node_props=None):
    if node_props is None:
        node_props = np.broadcast_to(1., csr.shape[0])
        node_props.flags.writeable = True
    return NBGraph(
            csr.indptr, csr.indices, csr.data,
            np.array(csr.shape, dtype=np.int32), node_props
            )


@numba.jit(nopython=True, cache=False)  # change this to True with Numba 1.0
def _build_paths(jgraph, indptr, indices, path_data, visited, degrees):
    indptr_i = 0
    indices_j = 0
    # first, process all nodes in a path to an endpoint or junction
    for node in range(jgraph.shape[0]):
        if degrees[node] > 2 or degrees[node] == 1:
            for neighbor in jgraph.neighbors(node):
                if not visited.edge(node, neighbor):
                    n_steps = _walk_path(
                            jgraph, node, neighbor, visited, degrees, indices,
                            path_data, indices_j
                            )
                    indptr[indptr_i + 1] = indptr[indptr_i] + n_steps
                    indptr_i += 1
                    indices_j += n_steps
    # everything else is by definition in isolated cycles
    for node in range(jgraph.shape[0]):
        if degrees[node] > 0:
            neighbor = jgraph.neighbors(node)[0]
            if not visited.edge(node, neighbor):
                n_steps = _walk_path(
                        jgraph, node, neighbor, visited, degrees, indices,
                        path_data, indices_j
                        )
                indptr[indptr_i + 1] = indptr[indptr_i] + n_steps
                indptr_i += 1
                indices_j += n_steps
    return indptr_i + 1, indices_j


@numba.jit(nopython=True, cache=False)  # change this to True with Numba 1.0
def _walk_path(
        jgraph, node, neighbor, visited, degrees, indices, path_data, startj
        ):
    indices[startj] = node
    start_node = node
    path_data[startj] = jgraph.node_properties[node]
    j = startj + 1
    while not visited.edge(node, neighbor):
        visited.set_edge(node, neighbor, True)
        visited.set_edge(neighbor, node, True)
        indices[j] = neighbor
        path_data[j] = jgraph.node_properties[neighbor]
        if degrees[neighbor] != 2 or neighbor == start_node:
            break
        n1, n2 = jgraph.neighbors(neighbor)
        nextneighbor = n1 if n1 != node else n2
        node, neighbor = neighbor, nextneighbor
        j += 1
    return j - startj + 1


def _build_skeleton_path_graph(graph):
    max_num_cycles = graph.indices.size // 4
    buffer_size_offset = max_num_cycles
    degrees = np.diff(graph.indptr)
    visited_data = np.zeros(graph.data.shape, dtype=bool)
    visited = NBGraphBool(
            graph.indptr, graph.indices, visited_data, graph.shape,
            np.broadcast_to(1., graph.shape[0])
            )
    endpoints = (degrees != 2)
    endpoint_degrees = degrees[endpoints]
    num_paths = np.sum(endpoint_degrees)
    path_indptr = np.zeros(num_paths + buffer_size_offset, dtype=int)
    # the number of points that we need to save to store all skeleton
    # paths is equal to the number of pixels plus the sum of endpoint
    # degrees minus one (since the endpoints will have been counted once
    # already in the number of pixels) *plus* the number of isolated
    # cycles (since each cycle has one index repeated). We don't know
    # the number of cycles ahead of time, but it is bounded by one quarter
    # of the number of points.
    n_points = (
            graph.indices.size + np.sum(endpoint_degrees - 1)
            + buffer_size_offset
            )
    path_indices = np.zeros(n_points, dtype=int)
    path_data = np.zeros(path_indices.shape, dtype=float)
    m, n = _build_paths(
            graph, path_indptr, path_indices, path_data, visited, degrees
            )
    paths = sparse.csr_matrix(
            (path_data[:n], path_indices[:n], path_indptr[:m]),
            shape=(m - 1, n)
            )
    return paths


def summarize(
        skel: Skeleton, *, value_is_height=False, find_main_branch=False
        ):
    """Compute statistics for every skeleton and branch in ``skel``.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object.
    value_is_height : bool
        Whether to consider the value of a float skeleton to be the "height"
        of the image. This can be useful e.g. when measuring lengths along
        ridges in AFM images.
    find_main_branch : bool, optional
        Whether to compute main branches. A main branch is defined as the
        longest shortest path within a skeleton. This step is very expensive
        as it involves computing the shortest paths between all pairs of branch
        endpoints, so it is off by default.

    Returns
    -------
    summary : pandas.DataFrame
        A summary of the branches including branch length, mean branch value,
        branch euclidean distance, etc.
    """
    summary = {}
    ndim = skel.coordinates.shape[1]
    _, skeleton_ids = csgraph.connected_components(skel.graph, directed=False)
    endpoints_src = skel.paths.indices[skel.paths.indptr[:-1]]
    endpoints_dst = skel.paths.indices[skel.paths.indptr[1:] - 1]
    summary['skeleton-id'] = skeleton_ids[endpoints_src]
    summary['node-id-src'] = endpoints_src
    summary['node-id-dst'] = endpoints_dst
    summary['branch-distance'] = skel.path_lengths()
    deg_src = skel.degrees[endpoints_src]
    deg_dst = skel.degrees[endpoints_dst]
    kind = np.full(deg_src.shape, 2)  # default: junction-to-junction
    kind[(deg_src == 1) | (deg_dst == 1)] = 1  # tip-junction
    kind[(deg_src == 1) & (deg_dst == 1)] = 0  # tip-tip
    kind[endpoints_src == endpoints_dst] = 3  # cycle
    summary['branch-type'] = kind
    summary['mean-pixel-value'] = skel.path_means()
    summary['stdev-pixel-value'] = skel.path_stdev()
    for i in range(ndim):  # keep loops separate for best insertion order
        summary[f'image-coord-src-{i}'] = skel.coordinates[endpoints_src, i]
    for i in range(ndim):
        summary[f'image-coord-dst-{i}'] = skel.coordinates[endpoints_dst, i]
    coords_real_src = skel.coordinates[endpoints_src] * skel.spacing
    for i in range(ndim):
        summary[f'coord-src-{i}'] = coords_real_src[:, i]
    if value_is_height:
        values_src = skel.pixel_values[endpoints_src]
        summary[f'coord-src-{ndim}'] = values_src
        coords_real_src = np.concatenate(
                [coords_real_src, values_src[:, np.newaxis]],
                axis=1,
                )  # yapf: ignore
    coords_real_dst = skel.coordinates[endpoints_dst] * skel.spacing
    for i in range(ndim):
        summary[f'coord-dst-{i}'] = coords_real_dst[:, i]
    if value_is_height:
        values_dst = skel.pixel_values[endpoints_dst]
        summary[f'coord-dst-{ndim}'] = values_dst
        coords_real_dst = np.concatenate(
                [coords_real_dst, values_dst[:, np.newaxis]],
                axis=1,
                )  # yapf: ignore
    summary['euclidean-distance'] = (
            np.sqrt((coords_real_dst - coords_real_src)**2
                    @ np.ones(ndim + int(value_is_height)))
            )
    df = pd.DataFrame(summary)

    if find_main_branch:
        # define main branch as longest shortest path within a single skeleton
        df['main'] = find_main_branches(df)
    return df


@numba.jit(nopython=True, nogil=True, cache=False)  # cache with Numba 1.0
def _compute_distances(graph, path_indptr, path_indices, distances):
    for i in range(len(distances)):
        start, stop = path_indptr[i:i + 2]
        path = path_indices[start:stop]
        distances[i] = _path_distance(graph, path)


@numba.jit(nopython=True, nogil=True, cache=False)  # cache with Numba 1.0
def _path_distance(graph, path):
    d = 0.
    n = len(path)
    for i in range(n - 1):
        u, v = path[i], path[i + 1]
        d += graph.edge(u, v)
    return d


def _mst_junctions(csmat):
    """Replace clustered pixels with degree > 2 by their minimum spanning tree.

    This function performs the operation in place.

    Parameters
    ----------
    csmat : NBGraph
        The input graph.
    pixel_indices : array of int
        The raveled index in the image of every pixel represented in csmat.
    spacing : float, or array-like of float, shape `len(shape)`, optional
        The spacing between pixels in the source image along each dimension.

    Returns
    -------
    final_graph : NBGraph
        The output csmat.
    """
    # make copy
    # mask out all degree < 3 entries
    # find MST
    # replace edges not in MST with zeros
    # use .eliminate_zeros() to get a new matrix
    csc_graph = csmat.tocsc()
    degrees = np.asarray(csmat.astype(bool).astype(int).sum(axis=0))
    non_junction = np.flatnonzero(degrees < 3)
    non_junction_column_start = csc_graph.indptr[non_junction]
    non_junction_column_end = csc_graph.indptr[non_junction + 1]
    for start, end in zip(non_junction_column_start, non_junction_column_end):
        csc_graph.data[start:end] = 0
    csr_graph = csc_graph.tocsr()
    non_junction_row_start = csr_graph.indptr[non_junction]
    non_junction_row_end = csr_graph.indptr[non_junction + 1]
    for start, end in zip(non_junction_row_start, non_junction_row_end):
        csr_graph.data[start:end] = 0
    csr_graph.eliminate_zeros()
    mst = csgraph.minimum_spanning_tree(csr_graph)
    non_tree_edges = csr_graph - (mst + mst.T)
    final_graph = csmat - non_tree_edges
    return final_graph


def distance_with_height(source_values, neighbor_values, distances):
    height_diff = source_values - neighbor_values
    return np.hypot(height_diff, distances)


def skeleton_to_csgraph(
        skel,
        *,
        spacing=1,
        value_is_height=False,
        ):
    """Convert a skeleton image of thin lines to a graph of neighbor pixels.

    Parameters
    ----------
    skel : array
        An input image in which every nonzero pixel is considered part of
        the skeleton, and links between pixels are determined by a full
        n-dimensional neighborhood.
    spacing : float, or array-like of float, shape `(skel.ndim,)`
        A value indicating the distance between adjacent pixels. This can
        either be a single value if the data has the same resolution along
        all axes, or it can be an array of the same shape as `skel` to
        indicate spacing along each axis.

    Other Parameters
    ----------------
    value_is_height : bool, optional
        If `True`, the pixel value at each point of the skeleton will be
        considered to be a height measurement, and this height will be
        incorporated into skeleton branch lengths. Used for analysis of
        atomic force microscopy (AFM) images.

    Returns
    -------
    graph : sparse.csr_matrix
        A graph of shape (Nnz, Nnz), where Nnz is the number of
        nonzero pixels in `skel`. The value graph[i, j] is the distance
        between adjacent pixels i and j. In a 2D image, that would be
        1 for immediately adjacent pixels and sqrt(2) for diagonally
        adjacent ones.
    pixel_coordinates : array of float
        An array of shape (Nnz, skel.ndim), mapping indices in `graph`
        to pixel coordinates in `skel`.
    """
    # ensure we have a bool image, since we later use it for bool indexing
    skel_im = skel
    skel_bool = skel.astype(bool)
    ndim = skel.ndim
    spacing = np.ones(ndim, dtype=float) * spacing

    if value_is_height:
        edge_func = distance_with_height
    else:
        edge_func = None

    graph, pixel_indices = pixel_graph(
            skel_im,
            mask=skel_bool,
            edge_function=edge_func,
            connectivity=ndim,
            spacing=spacing
            )

    graph = _mst_junctions(graph)
    pixel_coordinates = np.unravel_index(pixel_indices, skel.shape)
    return graph, pixel_coordinates


@numba.jit(nopython=True, cache=True)
def _csrget(indices, indptr, data, row, col):
    """Fast lookup of value in a scipy.sparse.csr_matrix format table.

    Parameters
    ----------
    indices, indptr, data : numpy arrays of int, int, float
        The CSR format data.
    row, col : int
        The matrix coordinates of the desired value.

    Returns
    -------
    dat: float
        The data value in the matrix.
    """
    start, end = indptr[row], indptr[row + 1]
    for i in range(start, end):
        if indices[i] == col:
            return data[i]
    return 0.


@numba.jit(nopython=True, cache=True)
def _csrset(indices, indptr, data, row, col, value):
    """Fast lookup and set of value in a scipy.sparse.csr_matrix format table.

    Parameters
    ----------
    indices, indptr, data : numpy arrays of int, int, float
        The CSR format data.
    row, col : int
        The matrix coordinates of the desired value.
    value : dtype
        The value to set in the matrix.

    Notes
    -----
    This function only sets values that already existed in the matrix.

    Returns
    -------
    success: bool
        Whether the data value was successfully written to the matrix.
    """
    start, end = indptr[row], indptr[row + 1]
    for i in range(start, end):
        if indices[i] == col:
            data[i] = value
            return True
    return False


def submatrix(M, idxs):
    """Return the outer-index product submatrix, `M[idxs, :][:, idxs]`.

    Parameters
    ----------
    M : scipy.sparse.spmatrix
        Input (square) matrix
    idxs : array of int
        The indices to subset. No index in `idxs` should exceed the
        number of rows of `M`.

    Returns
    -------
    Msub : scipy.sparse.spmatrix
        The subsetted matrix.

    Examples
    --------
    >>> Md = np.arange(16).reshape((4, 4))
    >>> M = sparse.csr_matrix(Md)
    >>> print(submatrix(M, [0, 2]).toarray())
    [[ 0  2]
     [ 8 10]]
    """
    Msub = M[idxs, :][:, idxs]
    return Msub


def make_degree_image(skeleton_image):
    """Create a array showing the degree of connectivity of each pixel.

    Parameters
    ----------
    skeleton_image : array
        An input image in which every nonzero pixel is considered part of
        the skeleton, and links between pixels are determined by a full
        n-dimensional neighborhood.

    Returns
    -------
    degree_image : array of int, same shape as skeleton_image
        An image containing the degree of connectivity of each pixel in the
        skeleton to neighboring pixels.
    """
    bool_skeleton = skeleton_image.astype(bool)
    degree_kernel = np.ones((3,) * bool_skeleton.ndim)
    degree_kernel[(1,) * bool_skeleton.ndim] = 0  # remove centre pixel
    if isinstance(bool_skeleton, np.ndarray):
        degree_image = ndi.convolve(
                bool_skeleton.astype(int),
                degree_kernel,
                mode='constant',
                ) * bool_skeleton
    # use dask image for any array other than a numpy array (which isn't
    # supported yet anyway)
    else:
        import dask.array as da
        from dask_image.ndfilters import convolve as dask_convolve
        if isinstance(bool_skeleton, da.Array):
            degree_image = bool_skeleton * dask_convolve(
                    bool_skeleton.astype(int), degree_kernel, mode='constant'
                    )
    return degree_image
