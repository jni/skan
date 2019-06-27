import numpy as np
import pandas as pd
from scipy import sparse, ndimage as ndi
from scipy.sparse import csgraph
from scipy import spatial
import numba

from .nputil import pad, raveled_steps_to_neighbors


## NBGraph and Numba-based implementation

csr_spec = [
    ('indptr', numba.int32[:]),
    ('indices', numba.int32[:]),
    ('data', numba.float64[:]),
    ('shape', numba.int32[:]),
    ('node_properties', numba.float64[:])
]

@numba.jitclass(csr_spec)
class NBGraph:
    def __init__(self, indptr, indices, data, shape, node_props):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape
        self.node_properties = node_props

    def edge(self, i, j):
        return _csrget(self.indices, self.indptr, self.data, i, j)

    def neighbors(self, row):
        loc, stop = self.indptr[row], self.indptr[row+1]
        return self.indices[loc:stop]

    @property
    def has_node_props(self):
        return self.node_properties.strides != (0,)


def csr_to_nbgraph(csr, node_props=None):
    if node_props is None:
        node_props = np.broadcast_to(1., csr.shape[0])
        node_props.flags.writeable = True
    return NBGraph(csr.indptr, csr.indices, csr.data,
                   np.array(csr.shape, dtype=np.int32), node_props)


def _pixel_graph(image, steps, distances, num_edges, height=None):
    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    data = np.empty(num_edges, dtype=float)
    if height is None:
        k = _write_pixel_graph(image, steps, distances, row, col, data)
    else:
        k = _write_pixel_graph_height(image, height, steps, distances,
                                      row, col, data)
    graph = sparse.coo_matrix((data[:k], (row[:k], col[:k]))).tocsr()
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

    Returns
    -------
    k : int
        The number of entries written to row, col, and data.

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
                if image[n] != 0 and image[n] != image[i]:
                    row[k] = image[i]
                    col[k] = image[n]
                    data[k] = distances[j]
                    k += 1
    return k

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

    Returns
    -------
    k : int
        The number of entries written to row, col, and data.

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
                if image[n] != 0 and image[n] != image[i]:
                    row[k] = image[i]
                    col[k] = image[n]
                    data[k] = np.sqrt(distances[j] ** 2 +
                                      (height[i] - height[n]) ** 2)
                    k += 1
    return k


@numba.jit(nopython=True, cache=False)  # change this to True with Numba 1.0
def _build_paths(jgraph, indptr, indices, path_data, visited, degrees):
    indptr_i = 0
    indices_j = 0
    # first, process all nodes in a path to an endpoint or junction
    for node in range(1, jgraph.shape[0]):
        if degrees[node] > 2 or degrees[node] == 1 and not visited[node]:
            for neighbor in jgraph.neighbors(node):
                if not visited[neighbor]:
                    n_steps = _walk_path(jgraph, node, neighbor, visited,
                                         degrees, indices, path_data,
                                         indices_j)
                    visited[node] = True
                    indptr[indptr_i + 1] = indptr[indptr_i] + n_steps
                    indptr_i += 1
                    indices_j += n_steps
    # everything else is by definition in isolated cycles
    for node in range(1, jgraph.shape[0]):
        if degrees[node] > 0:
            if not visited[node]:
                visited[node] = True
                neighbor = jgraph.neighbors(node)[0]
                n_steps = _walk_path(jgraph, node, neighbor, visited, degrees,
                                     indices, path_data, indices_j)
                indptr[indptr_i + 1] = indptr[indptr_i] + n_steps
                indptr_i += 1
                indices_j += n_steps
    return indptr_i + 1, indices_j


@numba.jit(nopython=True, cache=False)  # change this to True with Numba 1.0
def _walk_path(jgraph, node, neighbor, visited, degrees, indices, path_data,
               startj):
    indices[startj] = node
    path_data[startj] = jgraph.node_properties[node]
    j = startj + 1
    while degrees[neighbor] == 2 and not visited[neighbor]:
        indices[j] = neighbor
        path_data[j] = jgraph.node_properties[neighbor]
        n1, n2 = jgraph.neighbors(neighbor)
        nextneighbor = n1 if n1 != node else n2
        node, neighbor = neighbor, nextneighbor
        visited[node] = True
        j += 1
    indices[j] = neighbor
    path_data[j] = jgraph.node_properties[neighbor]
    visited[neighbor] = True
    return j - startj + 1


def _build_skeleton_path_graph(graph, *, _buffer_size_offset=None):
    if _buffer_size_offset is None:
        max_num_cycles = graph.indices.size // 4
        _buffer_size_offset = max_num_cycles
    degrees = np.diff(graph.indptr)
    visited = np.zeros(degrees.shape, dtype=bool)
    endpoints = (degrees != 2)
    endpoint_degrees = degrees[endpoints]
    num_paths = np.sum(endpoint_degrees)
    path_indptr = np.zeros(num_paths + _buffer_size_offset, dtype=int)
    # the number of points that we need to save to store all skeleton
    # paths is equal to the number of pixels plus the sum of endpoint
    # degrees minus one (since the endpoints will have been counted once
    # already in the number of pixels) *plus* the number of isolated
    # cycles (since each cycle has one index repeated). We don't know
    # the number of cycles ahead of time, but it is bounded by one quarter
    # of the number of points.
    n_points = (graph.indices.size + np.sum(endpoint_degrees - 1) +
                max_num_cycles)
    path_indices = np.zeros(n_points, dtype=int)
    path_data = np.zeros(path_indices.shape, dtype=float)
    m, n = _build_paths(graph, path_indptr, path_indices, path_data,
                        visited, degrees)
    paths = sparse.csr_matrix((path_data[:n], path_indices[:n],
                               path_indptr[:m]), shape=(m-1, n))
    return paths


class Skeleton:
    """Object to group together all the properties of a skeleton.

    In the text below, we use the following notation:

    - N: the number of points in the pixel skeleton,
    - ndim: the dimensionality of the skeleton
    - P: the number of paths in the skeleton (also the number of links in the
      junction graph).
    - J: the number of junction nodes
    - Sd: the sum of the degrees of all the junction nodes
    - [Nt], [Np], Nr, Nc: the dimensions of the source image

    Parameters
    ----------
    skeleton_image : array
        The input skeleton (1-pixel/voxel thick skeleton, all other values 0).

    Other Parameters
    ----------------
    spacing : float or array of float, shape ``(ndim,)``
        The scale of the pixel spacing along each axis.
    source_image : array of float, same shape as `skeleton_image`
        The image that `skeleton_image` represents / summarizes / was generated
        from. This is used to produce visualizations as well as statistical
        properties of paths.
    keep_images : bool
        Whether or not to keep the original input images. These can be useful
        for visualization, but they may take up a lot of memory.

    Attributes
    ----------
    graph : scipy.sparse.csr_matrix, shape (N + 1, N + 1)
        The skeleton pixel graph, where each node is a non-zero pixel in the
        input image, and each edge connects adjacent pixels. The graph is
        represented as an adjacency matrix in SciPy sparse matrix format. For
        more information see the ``scipy.sparse`` documentation as well as
        ``scipy.sparse.csgraph``. Note: pixel numbering starts at 1, so the
        shape of this matrix is ``(N + 1, N + 1)`` instead of ``(N, N)``.
    nbgraph : NBGraph
        A thin Numba wrapper around the ``csr_matrix`` format, this provides
        faster graph methods. For example, it is much faster to get a list of
        neighbors, or test for the presence of a specific edge.
    coordinates : array, shape (N, ndim)
        The image coordinates of each pixel in the skeleton.
    paths : scipy.sparse.csr_matrix, shape (P, N + 1)
        A csr_matrix where element [i, j] is on if node j is in path i. This
        includes path endpoints. The number of nonzero elements is N - J + Sd.
    n_paths : int
        The number of paths, P. This is redundant information given `n_paths`,
        but it is used often enough that it is worth keeping around.
    distances : array of float, shape (P,)
        The distance of each path.
    skeleton_image : array or None
        The input skeleton image. Only present if `keep_images` is True. Set to
        False to preserve memory.
    source_image : array or None
        The image from which the skeleton was derived. Only present if
        `keep_images` is True. This is useful for visualization.
    """
    def __init__(self, skeleton_image, *, spacing=1, source_image=None,
                 _buffer_size_offset=None, keep_images=True):
        graph, coords, degrees = skeleton_to_csgraph(skeleton_image,
                                                     spacing=spacing)
        if np.issubdtype(skeleton_image.dtype, np.float_):
            pixel_values = ndi.map_coordinates(skeleton_image, coords.T,
                                               order=3)
        else:
            pixel_values = None
        self.graph = graph
        self.nbgraph = csr_to_nbgraph(graph, pixel_values)
        self.coordinates = coords
        self.paths = _build_skeleton_path_graph(self.nbgraph,
                                    _buffer_size_offset=_buffer_size_offset)
        self.n_paths = self.paths.shape[0]
        self.distances = np.empty(self.n_paths, dtype=float)
        self._distances_initialized = False
        self.skeleton_image = None
        self.source_image = None
        self.degrees_image = degrees
        self.degrees = np.diff(self.graph.indptr)
        self.spacing = (np.asarray(spacing) if not np.isscalar(spacing)
                        else np.full(skeleton_image.ndim, spacing))
        if keep_images:
            self.skeleton_image = skeleton_image
            self.source_image = source_image

    def path(self, index):
        """Return the pixel indices of path number `index`.

        Parameters
        ----------
        index : int
            The desired path.

        Returns
        -------
        path : array of int
            The indices of the pixels belonging to the path, including
            endpoints.
        """
        # The below is equivalent to `self.paths[index].indices`, which is much
        # more elegant. However the below version is about 25x faster!
        # In [14]: %timeit mat[1].indices
        # 128 µs ± 421 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        # In [16]: %%timeit
        # ...: start, stop = mat.indptr[1:3]
        # ...: mat.indices[start:stop]
        # ...:
        # 5.05 µs ± 77.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        start, stop = self.paths.indptr[index:index+2]
        return self.paths.indices[start:stop]

    def path_coordinates(self, index):
        """Return the image coordinates of the pixels in the path.

        Parameters
        ----------
        index : int
            The desired path.

        Returns
        -------
        path_coords : array of float
            The (image) coordinates of points on the path, including endpoints.
        """
        path_indices = self.path(index)
        return self.coordinates[path_indices]

    def path_with_data(self, index):
        """Return pixel indices and corresponding pixel values on a path.

        Parameters
        ----------
        index : int
            The desired path.

        Returns
        -------
        path : array of int
            The indices of pixels on the path, including endpoints.
        data : array of float
            The values of pixels on the path.
        """
        start, stop = self.paths.indptr[index:index+2]
        return self.paths.indices[start:stop], self.paths.data[start:stop]

    def path_lengths(self):
        """Return the length of each path on the skeleton.

        Returns
        -------
        lengths : array of float
            The length of all the paths in the skeleton.
        """
        if not self._distances_initialized:
            _compute_distances(self.nbgraph, self.paths.indptr,
                               self.paths.indices, self.distances)
            self._distances_initialized = True
        return self.distances

    def paths_list(self):
        """List all the paths in the skeleton, including endpoints.

        Returns
        -------
        paths : list of array of int
            The list containing all the paths in the skeleton.
        """
        return [list(self.path(i)) for i in range(self.n_paths)]

    def path_means(self):
        """Compute the mean pixel value along each path.

        Returns
        -------
        means : array of float
            The average pixel value along each path in the skeleton.
        """
        sums = np.add.reduceat(self.paths.data, self.paths.indptr[:-1])
        lengths = np.diff(self.paths.indptr)
        return sums / lengths

    def path_stdev(self):
        """Compute the standard deviation of values along each path.

        Returns
        -------
        stdevs : array of float
            The standard deviation of pixel values along each path.
        """
        data = self.paths.data
        sumsq = np.add.reduceat(data * data, self.paths.indptr[:-1])
        lengths = np.diff(self.paths.indptr)
        means = self.path_means()
        return np.sqrt(np.clip(sumsq/lengths - means*means, 0, None))


def summarize(skel: Skeleton):
    """Compute statistics for every skeleton and branch in ``skel``.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object.

    Returns
    -------
    summary : pandas.DataFrame
        A summary of the branches including branch length, mean branch value,
        branch euclidean distance, etc.
    """
    summary = {}
    ndim = skel.coordinates.shape[1]
    _, skeleton_ids = csgraph.connected_components(skel.graph,
                                                   directed=False)
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
    coords_real_dst = skel.coordinates[endpoints_dst] * skel.spacing
    for i in range(ndim):
        summary[f'coord-dst-{i}'] = coords_real_dst[:, i]
    summary['euclidean-distance'] = (
            np.sqrt((coords_real_dst - coords_real_src)**2 @ np.ones(ndim))
    )
    df = pd.DataFrame(summary)
    return df


@numba.jit(nopython=True, nogil=True, cache=False)  # cache with Numba 1.0
def _compute_distances(graph, path_indptr, path_indices, distances):
    for i in range(len(distances)):
        start, stop = path_indptr[i:i+2]
        path = path_indices[start:stop]
        distances[i] = _path_distance(graph, path)


@numba.jit(nopython=True, nogil=True, cache=False)  # cache with Numba 1.0
def _path_distance(graph, path):
    d = 0.
    n = len(path)
    for i in range(n - 1):
        u, v = path[i], path[i+1]
        d += graph.edge(u, v)
    return d


def _uniquify_junctions(csmat, pixel_indices, junction_labels,
                        junction_centroids, *, spacing=1):
    """Replace clustered pixels with degree > 2 by a single "floating" pixel.

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
    junctions = np.unique(junction_labels)[1:]  # discard 0, background
    junction_centroids_real = junction_centroids * spacing
    for j, jloc in zip(junctions, junction_centroids_real):
        loc, stop = csmat.indptr[j], csmat.indptr[j+1]
        neighbors = csmat.indices[loc:stop]
        neighbor_locations = pixel_indices[neighbors]
        neighbor_locations *= spacing
        distances = np.sqrt(np.sum((neighbor_locations - jloc)**2, axis=1))
        csmat.data[loc:stop] = distances
    tdata = csmat.T.tocsr().data
    csmat.data = np.maximum(csmat.data, tdata)


def skeleton_to_csgraph(skel, *, spacing=1, value_is_height=False,
                        unique_junctions=True):
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
    unique_junctions : bool, optional
        If True, adjacent junction nodes get collapsed into a single
        conceptual node, with position at the centroid of all the connected
        initial nodes.

    Returns
    -------
    graph : sparse.csr_matrix
        A graph of shape (Nnz + 1, Nnz + 1), where Nnz is the number of
        nonzero pixels in `skel`. The value graph[i, j] is the distance
        between adjacent pixels i and j. In a 2D image, that would be
        1 for immediately adjacent pixels and sqrt(2) for diagonally
        adjacent ones.
    pixel_coordinates : array of float
        An array of shape (Nnz + 1, skel.ndim), mapping indices in `graph`
        to pixel coordinates in `degree_image` or `skel`.
    degree_image : array of int, same shape as skel
        An image where each pixel value contains the degree of its
        corresponding node in `graph`. This is useful to classify nodes.
    """
    height = pad(skel, 0.) if value_is_height else None
    # ensure we have a bool image, since we later use it for bool indexing
    skel = skel.astype(bool)
    ndim = skel.ndim
    spacing = np.ones(ndim, dtype=float) * spacing

    pixel_indices = np.concatenate(([[0.] * ndim],
                                    np.transpose(np.nonzero(skel))), axis=0)
    skelint = np.zeros(skel.shape, dtype=int)
    skelint[tuple(pixel_indices.T.astype(int))] = \
                                            np.arange(pixel_indices.shape[0])

    degree_kernel = np.ones((3,) * ndim)
    degree_kernel[(1,) * ndim] = 0  # remove centre pixel
    degree_image = ndi.convolve(skel.astype(int), degree_kernel,
                                mode='constant') * skel

    if unique_junctions:
        # group all connected junction nodes into "meganodes".
        junctions = degree_image > 2
        junction_ids = skelint[junctions]
        labeled_junctions, centroids = compute_centroids(junctions)
        labeled_junctions[junctions] = \
                                junction_ids[labeled_junctions[junctions] - 1]
        skelint[junctions] = labeled_junctions[junctions]
        pixel_indices[np.unique(labeled_junctions)[1:]] = centroids

    num_edges = np.sum(degree_image)  # *2, which is how many we need to store
    skelint = pad(skelint, 0)  # pad image to prevent looparound errors
    steps, distances = raveled_steps_to_neighbors(skelint.shape, ndim,
                                                  spacing=spacing)
    graph = _pixel_graph(skelint, steps, distances, num_edges, height)

    if unique_junctions:
        _uniquify_junctions(graph, pixel_indices,
                            labeled_junctions, centroids, spacing=spacing)
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
    graph : NBGraph
        A graph encoded identically to a SciPy sparse compressed sparse
        row matrix. See the documentation of `NBGraph` for details.
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
    n : int
        The number of pixels along the path followed (excluding the source).
    s : float
        The sum of the pixel values along the path followed (also excluding
        the source).
    deg : int
        The degree of `dest`.
    """
    d = graph.edge(source, step)
    s = 0.
    n = 0
    while degrees[step] == 2 and not visited[step]:
        n1, n2 = graph.neighbors(step)
        nextstep = n1 if n1 != source else n2
        source, step = step, nextstep
        d += graph.edge(source, step)
        visited[source] = True
        s += graph.node_properties[source]
        n += 1
    visited[step] = True
    return step, d, n, s, degrees[step]


@numba.jit(nopython=True, nogil=True)
def _branch_statistics_loop(jgraph, degrees, visited, result):
    num_results = 0
    for node in range(1, jgraph.shape[0]):
        if not visited[node]:
            if degrees[node] == 2:
                visited[node] = True
                left, right = jgraph.neighbors(node)
                id0, d0, n0, s0, deg0 = _expand_path(jgraph, node, left,
                                                     visited, degrees)
                if id0 == node:  # standalone cycle
                    id1, d1, n1, s1, deg1 = node, 0, 0, 0., 2
                    kind = 3
                else:
                    id1, d1, n1, s1, deg1 = _expand_path(jgraph, node, right,
                                                         visited, degrees)
                    kind = 2  # default: junction-to-junction
                    if deg0 == 1 and deg1 == 1:  # tip-tip
                        kind = 0
                    elif deg0 == 1 or deg1 == 1:  # tip-junct, tip-path impossible
                        kind = 1
                counts = n0 + n1 + 1
                values = s0 + s1 + jgraph.node_properties[node]
                result[num_results, :] = (float(id0), float(id1), d0 + d1,
                                          float(kind), values / counts)
                num_results += 1
            elif degrees[node] == 1:
                visited[node] = True
                neighbor = jgraph.neighbors(node)[0]
                id0, d0, n0, s0, deg0 = _expand_path(jgraph, node, neighbor,
                                                     visited, degrees)
                kind = 1 if deg0 > 2 else 0  # tip-junct / tip-tip
                counts = n0
                values = s0
                avg_value = np.nan if counts == 0 else values / counts
                result[num_results, :] = (float(node), float(id0), d0,
                                          float(kind), avg_value)
                num_results += 1
    return num_results


def branch_statistics(graph, pixel_values=None, *,
                      buffer_size_offset=0):
    """Compute the length and type of each branch in a skeleton graph.

    Parameters
    ----------
    graph : sparse.csr_matrix, shape (N, N)
        A skeleton graph.
    pixel_values : array of float, shape (N,)
        A value for each pixel in the graph. Used to compute total
        intensity statistics along each branch.
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
    branches : array of float, shape (N, {4, 5})
        An array containing branch endpoint IDs, length, and branch type.
        The types are:
        - tip-tip (0)
        - tip-junction (1)
        - junction-junction (2)
        - path-path (3) (This can only be a standalone cycle)
        Optionally, the last column contains the average pixel value
        along each branch (not including the endpoints).
    """
    jgraph = csr_to_nbgraph(graph, pixel_values)
    degrees = np.diff(graph.indptr)
    visited = np.zeros(degrees.shape, dtype=bool)
    endpoints = (degrees != 2)
    num_paths = np.sum(degrees[endpoints])
    result = np.zeros((num_paths + buffer_size_offset, 5), dtype=float)
    num_results = _branch_statistics_loop(jgraph, degrees, visited, result)
    num_columns = 5 if jgraph.has_node_props else 4
    return result[:num_results, :num_columns]


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


def summarise(image, *, spacing=1, using_height=False):
    """Compute statistics for every disjoint skeleton in `image`.

    **Note: this function is deprecated. Prefer** :func:`.summarize`.

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
    using_height : bool, optional
        If `True`, the pixel value at each point of the skeleton will be
        considered to be a height measurement, and this height will be
        incorporated into skeleton branch lengths, endpoint coordinates,
        and euclidean distances. Used for analysis of atomic force
        microscopy (AFM) images.

    Returns
    -------
    df : pandas DataFrame
        A data frame summarising the statistics of the skeletons in
        `image`.
    """
    ndim = image.ndim
    spacing = np.ones(ndim, dtype=float) * spacing
    g, coords_img, degrees = skeleton_to_csgraph(image, spacing=spacing,
                                                 value_is_height=using_height)
    num_skeletons, skeleton_ids = csgraph.connected_components(g,
                                                               directed=False)
    if np.issubdtype(image.dtype, np.float_) and not using_height:
        pixel_values = ndi.map_coordinates(image, coords_img.T, order=3)
        value_columns = ['mean pixel value']
        value_column_types = [float]
    else:
        pixel_values = None
        value_columns = []
        value_column_types = []
    stats = branch_statistics(g, pixel_values)
    indices0 = stats[:, 0].astype(int)
    indices1 = stats[:, 1].astype(int)
    coords_img0 = coords_img[indices0]
    coords_img1 = coords_img[indices1]
    coords_real0 = coords_img0 * spacing
    coords_real1 = coords_img1 * spacing
    if using_height:
        height_coords0 = ndi.map_coordinates(image, coords_img0.T, order=3)
        coords_real0 = np.column_stack((height_coords0, coords_real0))
        height_coords1 = ndi.map_coordinates(image, coords_img1.T, order=3)
        coords_real1 = np.column_stack((height_coords1, coords_real1))
    distances = np.sqrt(np.sum((coords_real0 - coords_real1)**2, axis=1))
    skeleton_id = skeleton_ids[stats[:, 0].astype(int)]
    table = np.column_stack((skeleton_id, stats, coords_img0, coords_img1,
                             coords_real0, coords_real1, distances))
    height_ndim = ndim if not using_height else (ndim + 1)
    columns = (['skeleton-id', 'node-id-0', 'node-id-1', 'branch-distance',
                'branch-type'] +
               value_columns +
               ['image-coord-src-%i' % i for i in range(ndim)] +
               ['image-coord-dst-%i' % i for i in range(ndim)] +
               ['coord-src-%i' % i for i in range(height_ndim)] +
               ['coord-dst-%i' % i for i in range(height_ndim)] +
               ['euclidean-distance'])
    column_types = ([int, int, int, float, int] + value_column_types +
                    2 * ndim * [int] +
                    2 * height_ndim * [float] +
                    [float])
    data_dict = {col: dat.astype(dtype)
                 for col, dat, dtype in zip(columns, table.T, column_types)}
    df = pd.DataFrame(data_dict)
    return df


def compute_centroids(image):
    """Find the centroids of all nonzero connected blobs in `image`.

    Parameters
    ----------
    image : ndarray
        The input image.

    Returns
    -------
    label_image : ndarray of int
        The input image, with each connected region containing a different
        integer label.

    Examples
    --------
    >>> image = np.array([[1, 0, 1, 0, 0, 1, 1],
    ...                   [1, 0, 0, 1, 0, 0, 0]])
    >>> labels, centroids = compute_centroids(image)
    >>> print(labels)
    [[1 0 2 0 0 3 3]
     [1 0 0 2 0 0 0]]
    >>> centroids
    array([[0.5, 0. ],
           [0.5, 2.5],
           [0. , 5.5]])
    """
    connectivity = np.ones((3,) * image.ndim)
    labeled_image = ndi.label(image, connectivity)[0]
    nz = np.nonzero(labeled_image)
    nzpix = labeled_image[nz]
    sizes = np.bincount(nzpix)
    coords = np.transpose(nz)
    grouping = np.argsort(nzpix)
    sums = np.add.reduceat(coords[grouping], np.cumsum(sizes)[:-1])
    means = sums / sizes[1:, np.newaxis]
    return labeled_image, means
