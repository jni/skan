import networkx as nx
import numpy as np
from skimage.draw import random_shapes
from skimage.morphology import skeletonize

tinycycle = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype=bool)


tinyline = np.array([0, 1, 1, 1, 0], dtype=bool)


skeleton0 = np.array([[0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1]], dtype=bool)


skeleton1 = np.array([[0, 1, 1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 0, 0, 1],
                      [0, 1, 1, 0, 1, 1, 0],
                      [1, 0, 0, 1, 0, 0, 0],
                      [1, 0, 0, 0, 1, 1, 1]], dtype=bool)


_zeros1 = np.zeros_like(skeleton1)
skeleton2 = np.concatenate((skeleton1, _zeros1), axis=1)
skeleton2 = np.concatenate((skeleton2, skeleton2[:, ::-1]), axis=0)

skeleton3d = np.array([[[1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]],
                       [[0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 1],
                        [1, 1, 0, 1, 0]],
                       [[0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]],
                       [[0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0]],
                       [[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1],
                        [1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1]]], dtype=bool)

topograph1d = np.array([3., 2., 3.])

skeleton4 = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1],
                      [0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0]], dtype=bool)

junction_first = np.array([[0, 1, 1, 1, 1],
                           [1, 1, 0, 0, 0],
                           [1, 0, 1, 0, 0],
                           [1, 0, 0, 1, 0],
                           [1, 0, 0, 0, 1]], dtype=bool)

skeletonlabel = np.array([[1, 1, 0, 0, 2, 2, 0],
                          [0, 0, 1, 0, 0, 0, 2],
                          [3, 0, 0, 1, 0, 0, 2],
                          [3, 0, 0, 1, 0, 0, 0],
                          [3, 0, 0, 0, 1, 1, 1],
                          [0, 3, 0, 0, 0, 1, 0]], dtype=int)


def _generate_random_skeleton(**extra_kwargs):
    """Generate random skeletons using skimage.draw's random_shapes."""
    kwargs = {"image_shape": (128, 128),
              "max_shapes": 20,
              "channel_axis": None,
              "shape": None,
              "allow_overlap": True}
    random_image, _ = random_shapes(**kwargs, **extra_kwargs)
    mask = random_image != 255
    return skeletonize(mask)

# Generate random skeletons:

# Skeleton with loop to be retained and side-branches
skeleton_loop1 = _generate_random_skeleton(rng=1, min_size=20)
# Skeleton with loop to be retained and side-branches
skeleton_loop2 = _generate_random_skeleton(rng=165103, min_size=60)
# Linear skeleton with lots of large side-branches, some forked
skeleton_linear1 = _generate_random_skeleton(rng=13588686514, min_size=20)
# Linear Skeleton with simple fork at one end
skeleton_linear2 = _generate_random_skeleton(rng=21, min_size=20)
# Linear Skeletons (i.e. multiple) with branches
skeleton_linear3 = _generate_random_skeleton(rng=894632511, min_size=20)

## Sample NetworkX Graphs...
# ...with no edge attributes
nx_graph = nx.Graph()
nx_graph.add_nodes_from([1, 2, 3])
# ...with edge attributes
nx_graph_edges = nx.Graph()
nx_graph_edges.add_nodes_from([1, 2, 3])
nx_graph_edges.add_edge(1, 2, **{"path": np.asarray([[4, 4]])})
