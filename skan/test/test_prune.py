import pytest

from skan._testdata import skeleton0
from skan import Skeleton


@pytest.mark.parametrize('branch_num', [0, 1, 2])
def test_pruning_comprehensive(branch_num):
    skeleton = Skeleton(skeleton0)
    pruned = skeleton.prune_paths([branch_num])
    print(pruned.skeleton_image.astype(int))
    assert pruned.n_paths == 1
