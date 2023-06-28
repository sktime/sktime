"""Testing sampling utilities."""

import numpy as np
import pytest

from sktime.datasets import load_unit_test
from sktime.datatypes import check_is_scitype
from sktime.utils._testing.deep_equals import deep_equals
from sktime.utils.sampling import random_partition, stratified_resample

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


def test_stratified_resample():
    """Test resampling returns valid data structure and maintains class distribution."""
    trainX, trainy = load_unit_test(split="TRAIN")
    testX, testy = load_unit_test(split="TEST")
    new_trainX, new_trainy, new_testX, new_testy = stratified_resample(
        trainX, trainy, testX, testy, 0
    )

    valid_train = check_is_scitype(new_trainX, scitype="Panel")
    valid_test = check_is_scitype(new_testX, scitype="Panel")
    assert valid_test and valid_train
    # count class occurrences
    unique_train, counts_train = np.unique(trainy, return_counts=True)
    unique_test, counts_test = np.unique(testy, return_counts=True)
    unique_train_new, counts_train_new = np.unique(new_trainy, return_counts=True)
    unique_test_new, counts_test_new = np.unique(new_testy, return_counts=True)
    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)
