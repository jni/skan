import numpy as np
import pandas as pd
from scipy import sparse, ndimage as ndi
from scipy.sparse import csgraph
import numba

from .nputil import pad, raveled_steps_to_neighbors


## CSGraph and Numba-based implementation

csr_spec = [
    ('indptr', numba.int32[:]),
    ('indices', numba.int32[:]),
    ('data', numba.float64[:]),
    ('shape', numba.int32[:])
]

@numba.jitclass(csr_spec)
class CSGraph:
    def __init__(self, indptr, indices, data, shape):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape

    @property
    def ndim(self):
        return self.shape.size

    def edge(self, i, j):
        return _csrget(self.indices, self.indptr, self.data, i, j)

    def neighbors(self, row):
        loc, stop = self.indptr[row], self.indptr[row+1]
        return self.indices[loc:stop]


def _pixel_graph(image, steps, distances, num_edges, height=None):
    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    data = np.empty(num_edges, dtype=float)
    if height is None:
        _write_pixel_graph(image, steps, distances, row, col, data)
    else:
        _write_pixel_graph_height(image, height, steps, distances,
                                  row, col, data)
    graph = sparse.coo_matrix((data, (row, col))).tocsr()
    return graph


@numba.jit(nopython=True, cache=True, nogil=True)
def _write_pixel_graph(image, steps, distances, row, col, data):
    """Step over `image` to build a graph of nonzero pixel neighbors.

    Parameters
    ----------
    image : int array
        The input image.
    steps : int array, shape (N,)
        The raveled index steps to find a pixel's neighbors in `image`.
    distances : float array, shape (N,)
        The euclidean distance from a pixel to its corresponding
        neighbor in `steps`.
    row : int array
        Output array to be filled with the "center" pixel IDs.
    col : int array
        Output array to be filled with the "neighbor" pixel IDs.
    data : float array
        Output array to be filled with the distances from center to
        neighbor pixels.

    Notes
    -----
    No size or bounds checking is performed. Users should ensure that
    - No index in `indices` falls on any edge of `image` (or the
      neighbor computation will fail or segfault).
    - The `steps` and `distances` arrays have the same shape.
    - The `row`, `col`, `data` are long enough to hold all of the
      edges.
    """
    image = image.ravel()
    n_neighbors = steps.size
    start_idx = np.max(steps)
    end_idx = image.size + np.min(steps)
    k = 0
    for i in range(start_idx, end_idx + 1):
        if image[i] != 0:
            for j in range(n_neighbors):
                n = steps[j] + i
                if image[n] != 0:
                    row[k] = image[i]
                    col[k] = image[n]
                    data[k] = distances[j]
                    k += 1

@numba.jit(nopython=True, cache=True, nogil=True)
def _write_pixel_graph_height(image, height, steps, distances, row, col, data):
    """Step over `image` to build a graph of nonzero pixel neighbors.

    Parameters
    ----------
    image : int array
        The input image.
    height : float array, same shape as `image`
        This is taken to be a height map along an additional
        dimension (in addition to the image dimensions), so the distance
        between two neighbors `i` and `n` separated by `j` is given by:

             `np.sqrt(distances[j]**2 + (height[i] - height[n])**2)`

    steps : int array, shape (N,)
        The raveled index steps to find a pixel's neighbors in `image`.
    distances : float array, shape (N,)
        The euclidean distance from a pixel to its corresponding
        neighbor in `steps`.
    row : int array
        Output array to be filled with the "center" pixel IDs.
    col : int array
        Output array to be filled with the "neighbor" pixel IDs.
    data : float array
        Output array to be filled with the distances from center to
        neighbor pixels.

    Notes
    -----
    No size or bounds checking is performed. Users should ensure that
    - No index in `indices` falls on any edge of `image` (or the
      neighbor computation will fail or segfault).
    - The `steps` and `distances` arrays have the same shape.
    - The `row`, `col`, `data` are long enough to hold all of the
      edges.
    """
    image = image.ravel()
    height = height.ravel()
    n_neighbors = steps.size
    start_idx = np.max(steps)
    end_idx = image.size + np.min(steps)
    k = 0
    for i in range(start_idx, end_idx + 1):
        if image[i] != 0:
            for j in range(n_neighbors):
                n = steps[j] + i
                if image[n] != 0:
                    row[k] = image[i]
                    col[k] = image[n]
                    data[k] = np.sqrt(distances[j] ** 2 +
                                      (height[i] - height[n]) ** 2)
                    k += 1


def skeleton_to_csgraph(skel, *, spacing=1):
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

    Returns
    -------
    graph : sparse.csr_matrix
        A graph of shape (Nnz + 1, Nnz + 1), where Nnz is the number of
        nonzero pixels in `skel`. The value graph[i, j] is the distance
        between adjacent pixels i and j. In a 2D image, that would be
        1 for immediately adjacent pixels and sqrt(2) for diagonally
        adjacent ones.
    pixel_indices : array of int
        An array of shape (Nnz + 1,), mapping indices in `graph` to
        raveled indices in `degree_image` or `skel`.
    degree_image : array of int, same shape as skel
        An image where each pixel value contains the degree of its
        corresponding node in `graph`. This is useful to classify nodes.
    """
    if np.issubdtype(skel.dtype, float):  # interpret float skels as height
        height = pad(skel, 0.)
    else:
        height = None
    skel = skel.astype(bool)  # ensure we have a bool image
                              # since we later use it for bool indexing
    spacing = np.ones(skel.ndim, dtype=float) * spacing

    ndim = skel.ndim
    pixel_indices = np.concatenate(([0], np.flatnonzero(skel)))
    skelint = np.zeros(skel.shape, int)
    skelint.ravel()[pixel_indices] = np.arange(pixel_indices.size + 1)
    skelint = pad(skelint, 0)

    degree_kernel = np.ones((3,) * ndim)
    degree_kernel.ravel()[3**ndim // 2] = 0  # remove centre pix
    degree_image = ndi.convolve(skel.astype(int), degree_kernel,
                                mode='constant') * skel
    num_edges = np.sum(degree_image)  # *2, which is how many we need to store
    steps, distances = raveled_steps_to_neighbors(skelint.shape, ndim,
                                                  spacing=spacing)
    graph = _pixel_graph(skelint, steps, distances, num_edges, height)
    return graph, pixel_indices, degree_image


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
    start, end = indptr[row], indptr[row+1]
    for i in range(start, end):
        if indices[i] == col:
            return data[i]
    return 0.


@numba.jit(nopython=True)
def _expand_path(graph, source, step, visited, degrees):
    """Walk a path on a graph until reaching a tip or junction.

    A path is a sequence of degree-2 nodes.

    Parameters
    ----------
    graph : CSGraph
        A graph encoded identically to a SciPy sparse compressed sparse
        row matrix. See the documentation of `CSGraph` for details.
    source : int
        The starting point of the walk. This must be a path node, or
        the function's behaviour is undefined.
    step : int
        The initial direction of the walk. Must be a neighbor of
        `source`.
    visited : array of bool
        An array mapping node ids to `False` (unvisited node) or `True`
        (previously visited node).
    degrees : array of int
        An array mapping node ids to their degrees in `graph`.

    Returns
    -------
    dest : int
        The tip or junction node at the end of the path.
    d : float
        The distance travelled from `source` to `dest`.
    deg : int
        The degree of `dest`.
    """
    d = graph.edge(source, step)
    while degrees[step] == 2 and not visited[step]:
        n1, n2 = graph.neighbors(step)
        nextstep = n1 if n1 != source else n2
        source, step = step, nextstep
        d += graph.edge(source, step)
        visited[source] = True
    return step, d, degrees[step]


def branch_statistics(graph, pixel_indices, degree_image, *,
                      buffer_size_offset=0):
    """Compute the length and type of each branch in a skeleton graph.

    Parameters
    ----------
    graph : sparse.csr_matrix, shape (N, N)
        A skeleton graph.
    pixel_indices : array of int, shape (N,)
        A map from rows/cols of `graph` to image coordinates.
    degree_image : array of int, shape (P, Q, ...)
        The image corresponding to the skeleton, where each value is
        its degree in `graph`.
    buffer_size_offset : int, optional
        The buffer size is given by the sum of the degrees of non-path
        nodes. This is usually 2x the amount needed, allowing room for
        extra cycles of path-only nodes. However, if the image consists
        *only* of such cycles, the buffer size will be 0, resulting in
        an error. Until a more sophisticated, expandable-buffer
        solution is implemented, you can manually set a bigger buffer
        size using this parameter.

    Returns
    -------
    branches : array of float, shape (N, 4)
        An array containing branch endpoint IDs, length, and branch type.
        The types are:
        - tip-tip (0)
        - tip-junction (1)
        - junction-junction (2)
        - path-path (3) (This can only be a standalone cycle)
    """
    jgraph = CSGraph(graph.indptr, graph.indices, graph.data,
                     np.array(graph.shape, np.int32))
    degree_image = degree_image.ravel()
    degrees = degree_image[pixel_indices]
    visited = np.zeros(pixel_indices.shape, dtype=bool)
    endpoints = (degrees != 2)
    num_paths = np.sum(degrees[endpoints])
    result = np.zeros((num_paths + buffer_size_offset, 4), dtype=float)
    num_results = 0
    for node in range(1, graph.shape[0]):
        if degrees[node] == 2 and not visited[node]:
            visited[node] = True
            left, right = jgraph.neighbors(node)
            id0, d0, deg0 = _expand_path(jgraph, node, left, visited, degrees)
            if id0 == node:  # standalone cycle
                id1, d1, deg1 = node, 0., 2
                kind = 3
            else:
                id1, d1, deg1 = _expand_path(jgraph, node, right, visited,
                                             degrees)
                kind = 2  # default: junction-to-junction
                if deg0 == 1 and deg1 == 1:  # tip-tip
                    kind = 0
                elif deg0 == 1 or deg1 == 1:  # tip-junct, tip-path impossible
                    kind = 1
            result[num_results, :] = id0, id1, d0 + d1, kind
            num_results += 1
    return result[:num_results]


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


def summarise(image, *, spacing=1):
    """Compute statistics for every disjoint skeleton in `image`.

    Parameters
    ----------
    image : array, shape (M, N, ..., P)
        N-dimensional array, where nonzero entries correspond to an
        object's single-pixel-wide skeleton. If the image is of type 'float',
        the values are taken to be the height at that pixel, which is used
        to compute the skeleton distances.
    spacing : float, or array-like of float, shape `(skel.ndim,)`
        A value indicating the distance between adjacent pixels. This can
        either be a single value if the data has the same resolution along
        all axes, or it can be an array of the same shape as `skel` to
        indicate spacing along each axis.

    Returns
    -------
    df : pandas DataFrame
        A data frame summarising the statistics of the skeletons in
        `image`.
    """
    ndim = image.ndim
    using_height = np.issubdtype(image.dtype, float)
    spacing = np.ones(ndim, dtype=float) * spacing
    g, pixels, degrees = skeleton_to_csgraph(image, spacing=spacing)
    coords_img = np.transpose(np.unravel_index(pixels, image.shape))
    num_skeletons, skeleton_ids = csgraph.connected_components(g,
                                                               directed=False)
    stats = branch_statistics(g, pixels, degree_image=degrees)
    indices0 = stats[:, 0].astype(int)
    indices1 = stats[:, 1].astype(int)
    coords_img0 = coords_img[indices0]
    coords_img1 = coords_img[indices1]
    coords_real0 = coords_img0 * spacing
    coords_real1 = coords_img1 * spacing
    if using_height:
        coords_real0 = np.column_stack((image.ravel()[pixels][indices0],
                                        coords_real0))
        coords_real1 = np.column_stack((image.ravel()[pixels][indices1],
                                        coords_real1))
    distances = np.sqrt(np.sum((coords_real0 - coords_real1)**2, axis=1))
    skeleton_id = skeleton_ids[stats[:, 0].astype(int)]
    table = np.column_stack((skeleton_id, stats, coords_img0, coords_img1,
                             coords_real0, coords_real1, distances))
    height_ndim = ndim if not using_height else (ndim + 1)
    columns = (['skeleton-id', 'node-id-0', 'node-id-1', 'branch-distance',
                'branch-type'] +
               ['img-coord-0-%i' % i for i in range(ndim)] +
               ['img-coord-1-%i' % i for i in range(ndim)] +
               ['coord-0-%i' % i for i in range(height_ndim)] +
               ['coord-1-%i' % i for i in range(height_ndim)] +
               ['euclidean-distance'])
    column_types = ([int, int, int, float, int] +
                    2 * ndim * [int] +
                    2 * height_ndim * [float] +
                    [float])
    data_dict = {col: dat.astype(dtype)
                 for col, dat, dtype in zip(columns, table.T, column_types)}
    df = pd.DataFrame(data_dict)
    return df
