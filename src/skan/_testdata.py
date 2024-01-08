import networkx as nx
import numpy as np
from skimage.draw import random_shapes
from skimage.morphology import skeletonize

tinycycle = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype=bool)


tinyline = np.array([[0, 1, 1, 1, 0]], dtype=bool)


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
skeleton2 = np.column_stack((skeleton1, _zeros1))
skeleton2 = np.row_stack((skeleton2, skeleton2[:, ::-1]))

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

# Generate a random skeletons, first is a skeleton with a closed loop with side branches
kwargs = {"image_shape": (128, 128),
          "max_shapes": 20,
          "channel_axis": None,
          "shape": None,
          "rng": 1,
          "allow_overlap": True,
          "min_size": 20}
# Skeleton with loop to be retained and side-branches
random_images, _ = random_shapes(**kwargs)
mask = np.where(random_images != 255, 1, 0)
skeleton_loop1 = skeletonize(mask)
# Skeleton with loop to be retained and side-branches
kwargs["rng"] = 165103
kwargs["min_size"] = 60
random_images, _ = random_shapes(**kwargs)
mask = np.where(random_images != 255, 1, 0)
skeleton_loop2 = skeletonize(mask)
# Linear skeleton with lots of large side-branches, some forked
kwargs["rng"] = 13588686514
kwargs["min_size"] = 20
random_images, _ = random_shapes(**kwargs)
mask = np.where(random_images != 255, 1, 0)
skeleton_linear1 = skeletonize(mask)
# Linear Skeleton with simple fork at one end
kwargs["rng"] = 21
kwargs["min_size"] = 20
random_images, _ = random_shapes(**kwargs)
mask = np.where(random_images != 255, 1, 0)
skeleton_linear2 = skeletonize(mask)
# Linear Skeletons (i.e. multiple) with branches
kwargs["rng"] = 894632511
kwargs["min_size"] = 20
random_images, _ = random_shapes(**kwargs)
mask = np.where(random_images != 255, 1, 0)
skeleton_linear3 = skeletonize(mask)

## Sample NetworkX Graphs...
# ...with no edge attributes
nx_graph = nx.Graph()
nx_graph.add_nodes_from([1, 2, 3])
# ...with edge attributes
nx_graph_edges = nx.Graph()
nx_graph_edges.add_nodes_from([1, 2, 3])
nx_graph_edges.add_edge(1, 2, **{"path": np.asarray([[4, 4]])})
