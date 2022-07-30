# -*- coding: utf-8 -*-
"""Testing sampling utilities."""

import pytest

from sktime.utils._testing.deep_equals import deep_equals
from sktime.utils.sampling import random_partition

NK_FIXTURES = [(10, 3), (15, 5), (19, 6), (3, 1), (1, 2)]
SEED_FIXTURES = [42, 0, 100, -5]


@pytest.mark.parametrize("n, k", NK_FIXTURES)
def test_partition(n, k):
    """Test that random_partition returns a disjoint partition."""
    part = random_partition(n, k)

    assert isinstance(part, list)
    assert all(isinstance(x, list) for x in part)
    assert all(isinstance(x, int) for y in part for x in y)

    low_size = n // k
    hi_size = low_size + 1
    assert all(len(x) == low_size or len(x) == hi_size for x in part)

    part_union = set()
    for x in part:
        part_union = part_union.union(x)
    assert set(range(n)) == part_union

    for i, x in enumerate(part):
        for j, y in enumerate(part):
            if i != j:
                assert len(set(x).intersection(y)) == 0


@pytest.mark.parametrize("seed", SEED_FIXTURES)
@pytest.mark.parametrize("n, k", NK_FIXTURES)
def test_seed(n, k, seed):
    """Test that seed is deterministic."""
    part = random_partition(n, k, seed)
    part2 = random_partition(n, k, seed)

    assert deep_equals(part, part2)
