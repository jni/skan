import numpy as np
from skimage import morphology

from .csr import (
        skeleton_to_csgraph,
        _build_skeleton_path_graph,
        csr_to_nbgraph,
        _compute_distances,
        )


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
