import numpy as np
from scipy import sparse
from scipy.spatial import distance_matrix
from skimage.graph import central_pixel
from skimage.util._map_array import ArrayMap
from .csr import summarize
import warnings


def _simplify_graph(skel):
    """Create junction graph from skeleton summary.

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

    edges = sparse.coo_matrix((distance, (src_relab, dst_relab)),
                              shape=(n_nodes, n_nodes))
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
