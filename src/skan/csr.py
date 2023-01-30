import numpy as np
import pandas as pd
from scipy import sparse, ndimage as ndi
from scipy.sparse import csgraph
from scipy.spatial import distance_matrix
from skimage import morphology
from skimage.graph import central_pixel
from skimage.util._map_array import map_array, ArrayMap
import numba
import warnings

from .nputil import _raveled_offsets_and_distances
from .summary_utils import find_main_branches


def _weighted_abs_diff(values0, values1, distances):
    """A default edge function for complete image graphs.

    A pixel graph on an image with no edge values and no mask is a very
    boring regular lattice, so we define a default edge weight to be the
    absolute difference between values *weighted* by the distance
    between them.

    Parameters
    ----------
    values0 : array
        The pixel values for each node.
    values1 : array
        The pixel values for each neighbor.
    distances : array
        The distance between each node and its neighbor.

    Returns
    -------
    edge_values : array of float
        The computed values: abs(values0 - values1) * distances.
    """
    return np.abs(values0 - values1) * distances


def pixel_graph(
        image, *, mask=None, edge_function=None, connectivity=1, spacing=None
        ):
    """Create an adjacency graph of pixels in an image.

    Pixels where the mask is True are nodes in the returned graph, and they are
    connected by edges to their neighbors according to the connectivity
    parameter. By default, the *value* of an edge when a mask is given, or when
    the image is itself the mask, is the euclidean distance betwene the pixels.

    However, if an int- or float-valued image is given with no mask, the value
    of the edges is the absolute difference in intensity between adjacent
    pixels, weighted by the euclidean distance.

    Parameters
    ----------
    image : array
        The input image. If the image is of type bool, it will be used as the
        mask as well.
    mask : array of bool
        Which pixels to use. If None, the graph for the whole image is used.
    edge_function : callable
        A function taking an array of pixel values, and an array of neighbor
        pixel values, and an array of distances, and returning a value for the
        edge. If no function is given, the value of an edge is just the
        distance.
    connectivity : int
        The square connectivity of the pixel neighborhood: the number of
        orthogonal steps allowed to consider a pixel a neigbor. See
        `scipy.ndimage.generate_binary_structure` for details.
    spacing : tuple of float
        The spacing between pixels along each axis.

    Returns
    -------
    graph : scipy.sparse.csr_matrix
        A sparse adjacency matrix in which entry (i, j) is 1 if nodes i and j
        are neighbors, 0 otherwise.
    nodes : array of int
        The nodes of the graph. These correspond to the raveled indices of the
        nonzero pixels in the mask.
    """
    if image.dtype == bool and mask is None:
        mask = image
    if mask is None and edge_function is None:
        mask = np.ones_like(image, dtype=bool)
        edge_function = _weighted_abs_diff

    # Strategy: we are going to build the (i, j, data) arrays of a scipy
    # sparse COO matrix, then convert to CSR (which is fast).
    # - grab the raveled IDs of the foreground (mask == True) parts of the
    #   image **in the padded space**.
    # - broadcast them together with the raveled offsets to their neighbors.
    #   This gives us for each foreground pixel a list of neighbors (that
    #   may or may not be selected by the mask.) (We also track the *distance*
    #   to each neighbor.)
    # - select "valid" entries in the neighbors and distance arrays by indexing
    #   into the mask, which we can do since these are raveled indices.
    # - use np.repeat() to repeat each source index according to the number
    #   of neighbors selected by the mask it has. Each of these repeated
    #   indices will be lined up with its neighbor, i.e. **this is the i
    #   array** of the COO format matrix.
    # - use the mask as a boolean index to get a 1D view of the selected
    #   neighbors. **This is the j array.**
    # - by default, the same boolean indexing can be applied to the distances
    #   to each neighbor, to give the **data array.** Optionally, a
    #   provided edge function can be computed on the pixel values and the
    #   distances to give a different value for the edges.
    # Note, we use map_array to map the raveled coordinates in the padded
    # image to the ones in the original image, and those are the returned
    # nodes.
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    nodes_padded = np.flatnonzero(padded)
    neighbor_offsets_padded, distances_padded = _raveled_offsets_and_distances(
            padded.shape, connectivity=connectivity, spacing=spacing
            )
    neighbors_padded = nodes_padded[:, np.newaxis] + neighbor_offsets_padded
    neighbor_distances_full = np.broadcast_to(
            distances_padded, neighbors_padded.shape
            )
    nodes = np.flatnonzero(mask)
    nodes_sequential = np.arange(nodes.size)
    # neighbors outside the mask get mapped to 0, which is a valid index,
    # BUT, they will be masked out in the next step.
    neighbors = map_array(neighbors_padded, nodes_padded, nodes)
    neighbors_mask = padded.reshape(-1)[neighbors_padded]
    num_neighbors = np.sum(neighbors_mask, axis=1)
    indices = np.repeat(nodes, num_neighbors)
    indices_sequential = np.repeat(nodes_sequential, num_neighbors)
    neighbor_indices = neighbors[neighbors_mask]
    neighbor_distances = neighbor_distances_full[neighbors_mask]
    neighbor_indices_sequential = map_array(
            neighbor_indices, nodes, nodes_sequential
            )
    if edge_function is None:
        data = neighbor_distances
    else:
        image_r = image.reshape(-1)
        data = edge_function(
                image_r[indices], image_r[neighbor_indices], neighbor_distances
                )
    m = nodes_sequential.size
    mat = sparse.coo_matrix(
            (data, (indices_sequential, neighbor_indices_sequential)),
            shape=(m, m)
            )
    graph = mat.tocsr()
    return graph, nodes


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


def _pixel_graph(image, steps, distances, num_edges, height=None):
    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    data = np.empty(num_edges, dtype=float)
    if height is None:
        k = _write_pixel_graph(image, steps, distances, row, col, data)
    else:
        k = _write_pixel_graph_height(
                image, height, steps, distances, row, col, data
                )
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
                    data[k] = np.sqrt(
                            distances[j]**2 + (height[i] - height[n])**2
                            )
                    k += 1
    return k


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
            graph.indices.size + np.sum(np.maximum(0, endpoint_degrees - 1))
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
    value_is_height : bool
        Whether to consider the value of a float skeleton to be the "height"
        of the image. This can be useful e.g. when measuring lengths along
        ridges in AFM images.

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
        skeleton_pixel_id i -> coordinates[i]
        The image coordinates of each pixel in the skeleton.
        Some values in this matrix are non-sensical — you should only access
        them from node ids.
    paths : scipy.sparse.csr_matrix, shape (P, N + 1)
        A csr_matrix where element [i, j] is on if node j is in path i. This
        includes path endpoints. The number of nonzero elements is N - J + Sd.
    n_paths : int
        The number of paths, P. This is redundant information given `n_paths`,
        but it is used often enough that it is worth keeping around.
    distances : array of float, shape (P,)
        The distance of each path. Note: not initialized until `path_lengths()`
        is called on the skeleton; use path_lengths() instead
    skeleton_image : array or None
        The input skeleton image. Only present if `keep_images` is True. Set to
        False to preserve memory.
    source_image : array or None
        The image from which the skeleton was derived. Only present if
        `keep_images` is True. This is useful for visualization.
    """
    def __init__(
            self,
            skeleton_image,
            *,
            spacing=1,
            source_image=None,
            keep_images=True,
            value_is_height=False,
            ):
        graph, coords = skeleton_to_csgraph(
                skeleton_image,
                spacing=spacing,
                value_is_height=value_is_height,
                )
        if np.issubdtype(skeleton_image.dtype, np.float_):
            self.pixel_values = skeleton_image[coords]
        else:
            self.pixel_values = None
        self.graph = graph
        self.nbgraph = csr_to_nbgraph(graph, self.pixel_values)
        self.coordinates = np.transpose(coords)
        self.paths = _build_skeleton_path_graph(self.nbgraph)
        self.n_paths = self.paths.shape[0]
        self.distances = np.empty(self.n_paths, dtype=float)
        self._distances_initialized = False
        self.skeleton_image = None
        self.source_image = None
        self.degrees = np.diff(self.graph.indptr)
        self.spacing = (
                np.asarray(spacing) if not np.isscalar(spacing) else
                np.full(skeleton_image.ndim, spacing)
                )
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
        start, stop = self.paths.indptr[index:index + 2]
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
        start, stop = self.paths.indptr[index:index + 2]
        return self.paths.indices[start:stop], self.paths.data[start:stop]

    def path_lengths(self):
        """Return the length of each path on the skeleton.

        Returns
        -------
        lengths : array of float
            The length of all the paths in the skeleton.
        """
        if not self._distances_initialized:
            _compute_distances(
                    self.nbgraph, self.paths.indptr, self.paths.indices,
                    self.distances
                    )
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

    def path_label_image(self):
        """Image like self.skeleton_image with path_ids as values.

        Returns
        -------
        label_image : array of ints
            Image of the same shape as self.skeleton_image where each pixel
            has the value of its branch id + 1.
        """
        image_out = np.zeros(self.skeleton_image.shape, dtype=int)
        for i in range(self.n_paths):
            coords_to_wipe = self.path_coordinates(i)
            coords_idxs = tuple(np.round(coords_to_wipe).astype(int).T)
            image_out[coords_idxs] = i + 1
        return image_out

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

    def prune_paths(self, indices) -> 'Skeleton':
        # warning: slow
        image_cp = np.copy(self.skeleton_image)
        for i in indices:
            pixel_ids_to_wipe = self.path(i)
            junctions = self.degrees[pixel_ids_to_wipe] > 2
            pixel_ids_to_wipe = pixel_ids_to_wipe[~junctions]
            coords_to_wipe = self.coordinates[pixel_ids_to_wipe]
            coords_idxs = tuple(np.round(coords_to_wipe).astype(int).T)
            image_cp[coords_idxs] = 0
        # optional cleanup:
        new_skeleton = morphology.skeletonize(image_cp.astype(bool)) * image_cp
        return Skeleton(
                new_skeleton,
                spacing=self.spacing,
                source_image=self.source_image,
                )

    def __array__(self, dtype=None):
        """Array representation of the skeleton path labels."""
        return self.path_label_image()


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


def _simplify_graph(skel):
    """Iterative removal of all nodes of degree 2 while reconnecting their
    edges.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object containing graph to be simplified.

    Returns
    -------
    simp_csgraph : scipy.sparse.csr_matrix
        A sparse adjacency matrix of the simplified graph.
    reduced_nodes : tuple of int
        The index nodes of original graph in simplified graph.
    """
    if np.sum(skel.degrees > 2) == 0:  # no junctions
        # don't reduce
        return skel.graph, np.arange(skel.graph.shape[0])

    summary = summarize(skel)
    src = np.asarray(summary['node-id-src'])
    dst = np.asarray(summary['node-id-dst'])
    distance = np.asarray(summary['branch-distance'])

    # to reduce the size of simplified graph
    nodes = np.unique(np.append(src, dst))
    n_nodes = len(nodes)
    nodes_sequential = np.arange(n_nodes)

    fw_map = ArrayMap(nodes, nodes_sequential)
    inv_map = ArrayMap(nodes_sequential, nodes)

    src_relab, dst_relab = fw_map[src], fw_map[dst]

    edges = sparse.coo_matrix(
            (distance, (src_relab, dst_relab)),
            shape=(n_nodes, n_nodes)
            )
    dir_csgraph = edges.tocsr()
    simp_csgraph = dir_csgraph + dir_csgraph.T  # make undirected

    reduced_nodes = inv_map[np.arange(simp_csgraph.shape[0])]

    return simp_csgraph, reduced_nodes


def _fast_graph_center_idx(skel):
    """Accelerated graph center finding using simplified graph.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object containing graph whose center is to be found.

    Returns
    -------
    original_center_idx : int
        The index of central node of graph.
    """
    simp_csgraph, reduced_nodes = _simplify_graph(skel)
    simp_center_idx, _ = central_pixel(simp_csgraph)
    original_center_idx = reduced_nodes[simp_center_idx]

    return original_center_idx


def _normalize_shells(shells, *, center, skeleton_coordinates, spacing):
    """Normalize shells from any format allowed by `sholl_analysis` to radii.

    Parameters
    ----------
    shells : int or sequence of floats, or None
        If an int, it is used as number of evenly spaced concentric shells. If
        an array of floats, it is used directly as the different shell radii in
        real world units. If None, the number of evenly spaced concentric
        shells is automatically calculated.
    center : (D,) array of float
        The scaled coordinates of the center point for Sholl analysis.
    skeleton_coordinates : (N, D) array of float
        The scaled coordinates of skeleton pixels. Used when shells is None or
        int.
    spacing : (D,) array of float
        The pixel/voxel spacing of the skeleton data.

    Returns
    -------
    radii : array of float
        The computed and normalized shell radii.
    """
    if isinstance(shells, (list, tuple, np.ndarray)):
        shell_radii = np.asarray(shells)
    else:  # shells is int, number of shells, or None
        # Find max euclidean distance from center to all nodes
        distances = np.linalg.norm(skeleton_coordinates - center, axis=1)
        start_radius = 0
        end_radius = np.max(distances)  # largest possible radius
        if shells is None:
            stepsize = np.linalg.norm(spacing)
        else:  # scalar
            stepsize = (end_radius-start_radius) / shells
        epsilon = np.finfo(np.float32).eps
        shell_radii = np.arange(start_radius, end_radius + epsilon, stepsize)
    if (sp := np.linalg.norm(spacing)) > (sh := np.min(np.diff(shell_radii))):
        warnings.warn(
                'This implementation of Sholl analysis may not be accurate if '
                'the spacing between shells is smaller than the (diagonal) '
                f'voxel spacing. The given voxel spacing is {sp}, and the '
                f'smallest shell spacing is {sh}.',
                stacklevel=2
                )
    return shell_radii


def sholl_analysis(skeleton, center=None, shells=None):
    """Sholl Analysis for Skeleton object.

    Parameters
    ----------
    skeleton : skan.csr.Skeleton
        A Skeleton object.
    center : array-like of float or None, optional
        Scaled coordinates of a point on the skeleton to use as the center
        from which the concentric shells are computed. If None, the
        geodesic center of skeleton is chosen.
    shells : int or array of floats or None, optional
        If an int, it is used as number of evenly spaced concentric shells. If
        an array of floats, it is used directly as the different shell radii in
        real world units. If None, the number of evenly spaced concentric
        shells is automatically calculated.

    Returns
    -------
    center : array of float
        The scaled coordinates in real world units of the center of the shells.
        (This might be provided as input, but it might also have been computed
        within this function, and that computation is expensive, so we return
        it just in case.)
    shell_radii : array of float
        Radii in real world units for concentric shells used for analysis.
    intersection_counts : array of int
        Number of intersections for corresponding shell radii.
    """
    if center is None:
        # By default, find the geodesic center of the graph
        center_idx = _fast_graph_center_idx(skeleton)
        center = skeleton.coordinates[center_idx] * skeleton.spacing
    else:
        center = np.asarray(center)

    scaled_coords = skeleton.coordinates * skeleton.spacing

    shell_radii = _normalize_shells(
            shells,
            center=center,
            skeleton_coordinates=scaled_coords,
            spacing=skeleton.spacing,
            )

    edges = skeleton.graph.tocoo()
    coords0 = scaled_coords[edges.row]
    coords1 = scaled_coords[edges.col]
    d0 = distance_matrix(coords0, [center]).ravel()
    d1 = distance_matrix(coords1, [center]).ravel()
    bins0 = np.digitize(d0, shell_radii)
    bins1 = np.digitize(d1, shell_radii)
    crossings = bins0 != bins1
    shells = np.minimum(bins0[crossings], bins1[crossings])
    # we divide by 2 because the graph is undirected, so each edge appears
    # twice in the matrix
    intersection_counts = np.bincount(shells, minlength=len(shell_radii)) // 2

    return center, shell_radii, intersection_counts
