import numpy as np
import pandas as pd
from scipy import sparse, ndimage as ndi
from scipy.sparse import csgraph
import numba

from .nputil import pad, raveled_steps_to_neighbors


## CSGraph and Numba-based implementation



@numba.jit(nopython=True, cache=True, nogil=True)
def write_pixel_graph(image, steps, distances, row, col, data):
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
    start_idx = steps.size
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


def skeleton_to_csgraph(skel):
    """Convert a skeleton image of thin lines to a graph of neighbor pixels.

    Parameters
    ----------
    skel : array
        An input image in which every nonzero pixel is considered part of
        the skeleton, and links between pixels are determined by a full
        n-dimensional neighborhood.

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
    skel = skel.astype(bool)  # ensure we have a bool image
    # since we later use it for bool indexing
    ndim = skel.ndim
    pixel_indices = np.concatenate(([0], np.flatnonzero(skel)))
    skelint = np.zeros(skel.shape, int)
    skelint.ravel()[pixel_indices] = np.arange(pixel_indices.size + 1)
    skelint = pad(skelint, 0)

    degree_kernel = np.ones((3,) * ndim)
    degree_kernel.ravel()[3**ndim // 2] = 0  # remove centre pix
    degree_image = ndi.convolve(skel.astype(int), degree_kernel,
                                mode='constant')
    num_edges = np.sum(degree_image)  # *2, which is how many we need to store
    row, col = np.zeros(num_edges, dtype=int), np.zeros(num_edges, dtype=int)
    data = np.zeros(num_edges, dtype=float)
    steps, distances = raveled_steps_to_neighbors(skelint.shape, ndim)
    write_pixel_graph(skelint, steps, distances, row, col, data)
    graph = sparse.coo_matrix((data, (row, col))).tocsr()
    return graph, pixel_indices, degree_image


@numba.jit(nopython=True, cache=True)
def _csrget(indices, indptr, data, row, col):
    start, end = indptr[row], indptr[row+1]
    for i in range(start, end):
        if indices[i] == col:
            return data[i]
    return 0.


@numba.jit(nopython=True, cache=True)
def _expand_path(indices, indptr, data, source, step, visited, degrees):
    d = _csrget(indices, indptr, data, source, step)
    while degrees[step] == 2 and not visited[step]:
        loc = indptr[step]
        n1, n2 = indices[loc], indices[loc+1]
        nextstep = n1 if n1 != source else n2
        source, step = step, nextstep
        d += _csrget(indices, indptr, data, source, step)
        visited[source] = True
    return step, d, degrees[step]


def branch_statistics(graph, pixel_indices, degree_image):
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

    Returns
    -------
    branches : array of float, shape (N, 4)
        An array containing branch endpoint IDs, length, and branch type.
        The types are:
        - tip-tip (0)
        - tip-junction (1)
        - junction-junction (2)
    """
    degree_image = degree_image.ravel()
    degrees = degree_image[pixel_indices]
    visited = np.zeros(pixel_indices.shape, dtype=bool)
    endpoints = (degrees != 2)
    num_paths = np.sum(degrees[endpoints])
    result = np.zeros((num_paths, 4), dtype=float)
    num_results = 0
    num_cycles = 0
    for node in range(1, graph.shape[0]):
        if degrees[node] == 2 and not visited[node]:
            visited[node] = True
            loc = graph.indptr[node]
            left, right = graph.indices[loc:loc+2]
            id0, d0, deg0 = _expand_path(graph.indices, graph.indptr,
                                         graph.data, node, left, visited,
                                         degrees)
            id1, d1, deg1 = _expand_path(graph.indices, graph.indptr,
                                         graph.data, node, right, visited,
                                         degrees)
            kind = 2  # default: junction-to-junction
            if deg0 == 1 and deg1 == 1:  # tip-tip
                kind = 0
            elif deg0 == 1 or deg1 == 1:  # tip-junction, tip-path impossible
                kind = 1
            elif deg0 == 2:  # must be a cycle
                num_cycles += 1
                continue
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
    >>> submatrix(M, [0, 2]).toarray()
    array([[ 0,  2],
           [ 8, 10]], dtype=int64)
    """
    Msub = M[idxs, :][:, idxs]
    return Msub


def summarise(skelimage):
    ndim = skelimage.ndim
    g, pixels, degrees = skeleton_to_csgraph(skelimage)
    coords = np.transpose(np.unravel_index(pixels, skelimage.shape))
    num_skeletons, skeleton_ids = csgraph.connected_components(g,
                                                               directed=False)
    stats = branch_statistics(g, pixels, degree_image=degrees)
    coords0 = coords[stats[:, 0].astype(int)]
    coords1 = coords[stats[:, 1].astype(int)]
    distances = np.sqrt(np.sum((coords0 - coords1)**2, axis=1))
    skeleton_id = skeleton_ids[stats[:, 0].astype(int)]
    table = np.column_stack((skeleton_id, stats,
                             coords0, coords1, distances))
    columns = (['skeleton-id', 'node-id-0', 'node-id-1', 'branch-distance',
                'branch-type'] +
               ['coord-0-%i' % i for i in range(ndim)] +
               ['coord-1-%i' % i for i in range(ndim)] +
               ['euclidean-distance'])
    column_types = [int, int, int, float, int] + 2*ndim*[int] + [float]
    data_dict = {col: dat.astype(dtype)
                 for col, dat, dtype in zip(columns, table.T, column_types)}
    df = pd.DataFrame(data_dict)
    return df
