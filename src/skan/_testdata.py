import os

import networkx as nx
import numpy as np
import zarr

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


# Skeletons used to test (iterative) pruning.
#   skeleton_loop1, skeleton_loop2     -- closed loops with side branches
#   skeleton_linear1                   -- linear, many side branches + a small
#                                         spurious loop
#   skeleton_linear2                   -- linear with a simple fork at one end
#   skeleton_linear3                   -- multiple linear skeletons with branches
_PRUNING_ZARR = os.path.join(
        os.path.dirname(__file__), 'test', 'data', 'pruning_skeletons.zarr.zip'
        )


def _load_pruning_skeletons():
    store = zarr.storage.ZipStore(_PRUNING_ZARR, mode='r')
    try:
        group = zarr.open_group(store=store, mode='r')
        names = (
                'skeleton_loop1', 'skeleton_loop2', 'skeleton_linear1',
                'skeleton_linear2', 'skeleton_linear3'
                )
        return {name: np.asarray(group[name]) for name in names}
    finally:
        store.close()


_pruning_skeletons = _load_pruning_skeletons()
skeleton_loop1 = _pruning_skeletons['skeleton_loop1']
skeleton_loop2 = _pruning_skeletons['skeleton_loop2']
skeleton_linear1 = _pruning_skeletons['skeleton_linear1']
skeleton_linear2 = _pruning_skeletons['skeleton_linear2']
skeleton_linear3 = _pruning_skeletons['skeleton_linear3']

## Sample NetworkX Graphs...
# ...with no edge attributes
nx_graph = nx.Graph()
nx_graph.add_nodes_from([1, 2, 3])
# ...with edge attributes
nx_graph_edges = nx.Graph()
nx_graph_edges.add_nodes_from([1, 2, 3])
nx_graph_edges.add_edge(1, 2, **{"path": np.asarray([[4, 4]])})
