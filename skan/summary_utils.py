import networkx as nx
import numpy as np
from pandas import DataFrame
import toolz as tz


def find_main_branches(summary: DataFrame) -> np.ndarray:
    """Predict the extent of branching.

    Parameters
    ----------
    summary : pd.DataFrame
        The summary table of the skeleton to analyze.
        This must contain: ['node-id-src', 'node-id-dst', 'branch-distance']

    Returns
    -------
    is_main: array
       True if the index-matched path is the longest shortest path of the
       skeleton
    """
    is_main = np.zeros(summary.shape[0], dtype=bool)
    us = summary['node-id-src']
    vs = summary['node-id-dst']
    ws = summary['branch-distance']

    edge2idx = {(u, v): i for i, (u, v) in enumerate(zip(us, vs))}

    edge2idx.update({(v, u): i for i, (u, v) in enumerate(zip(us, vs))})

    g = nx.Graph()

    g.add_weighted_edges_from(zip(us, vs, ws))

    for conn in nx.connected_components(g):
        curr_val = 0
        curr_pair = None
        h = g.subgraph(conn)
        p = dict(nx.all_pairs_dijkstra_path_length(h))
        for src in p:
            for dst in p[src]:
                val = p[src][dst]
                if (val is not None and np.isfinite(val) and val >= curr_val):
                    curr_val = val
                    curr_pair = (src, dst)
        for i, j in tz.sliding_window(2, nx.shortest_path(h,
                                                          source=curr_pair[0],
                                                          target=curr_pair[1],
                                                          weight='weight')):
            is_main[edge2idx[(i, j)]] = 1

    return is_main
