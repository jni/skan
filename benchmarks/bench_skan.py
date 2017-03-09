import os

from contextlib import contextmanager
from collections import OrderedDict
from time import process_time

import numpy as np

from skan import csr

rundir = os.path.dirname(__file__)


@contextmanager
def timer():
    time = []
    t0 = process_time()
    yield time
    t1 = process_time()
    time.append(t1 - t0)


def bench_suite():
    times = OrderedDict()
    skeleton = np.load(os.path.join(rundir, 'infected3.npz'))['skeleton']
    with timer() as t_build_graph:
        g, indices, degrees = csr.skeleton_to_csgraph(skeleton,
                                                      spacing=2.24826)
    times['build graph'] = t_build_graph[0]
    with timer() as t_build_graph2:
        g, indices, degrees = csr.skeleton_to_csgraph(skeleton,
                                                      spacing=2.24826)
    times['build graph again'] = t_build_graph2[0]
    with timer() as t_stats:
        stats = csr.branch_statistics(g)
    times['compute statistics'] = t_stats[0]
    with timer() as t_stats2:
        stats = csr.branch_statistics(g)
    times['compute statistics again'] = t_stats2[0]
    with timer() as t_summary:
        summary = csr.summarise(skeleton)
    times['compute per-skeleton statistics'] = t_summary[0]
    return times


def print_bench_results(times=None, memory=None):
    if times is not None:
        print('Timing results:')
        for key in times:
            print('--- ', key, '%.3f s' % times[key])
    if memory is not None:
        print('Memory results:')
        for key in memory:
            print('--- ', key, '%.3f MB' % (memory[key] / 1e6))


if __name__ == '__main__':
    times = bench_suite()
    print_bench_results(times)
