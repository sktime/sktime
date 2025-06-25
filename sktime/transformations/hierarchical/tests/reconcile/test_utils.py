import pandas as pd
import pytest

from sktime.transformations.hierarchical.reconcile._utils import (
    _promote_hierarchical_indexes,
    _recursively_propagate_topdown,
)


@pytest.mark.parametrize(
    "idx_tuple, expected_result",
    [
        # Case 1: No "__total" in the tuple, add it to the last level
        (("region", "store", "product"), ("region", "store", "__total")),
        # Case 2: "__total" already present at the last level, move up one level
        (("region", "store", "__total"), ("region", "__total", "__total")),
        # Case 3: "__total" at the first level, keep unchanged
        (("__total", "__total", "__total"), ("__total", "__total", "__total")),
        # Case 5: "__total" in the middle of the tuple
        (("region", "__total", "__total"), ("__total", "__total", "__total")),
    ],
)
def test_walk_up_hierarchical_levels(idx_tuple, expected_result):
    result = _promote_hierarchical_indexes(idx_tuple)
    assert result == expected_result, f"Failed for input: {idx_tuple}"


def example_data_for_topdown_all_ones():
    """3-level MultiIndex, all values = 1.0 (should remain unchanged)."""
    index = pd.MultiIndex.from_tuples(
        [
            ("__total", "__total", "__total", pd.Period("2020-01-01")),
            ("regionA", "__total", "__total", pd.Period("2020-01-01")),
            ("regionA", "storeB", "__total", pd.Period("2020-01-01")),
            ("regionA", "storeB", "p1", pd.Period("2020-01-01")),
            ("regionA", "storeB", "p2", pd.Period("2020-01-01")),
            ("regionB", "__total", "__total", pd.Period("2020-01-01")),
            ("regionB", "storeA", "__total", pd.Period("2020-01-01")),
            ("regionB", "storeA", "p1", pd.Period("2020-01-01")),
            ("regionB", "storeA", "p2", pd.Period("2020-01-01")),
        ],
        names=["region", "store", "product", "period"],
    )
    X = pd.Series(1.0, index=index)
    return X, X


def example_data_for_topdown_varied():
    """3-level MultiIndex, varied values for more thorough test."""
    index = pd.MultiIndex.from_tuples(
        [
            ("__total", "__total", "__total", pd.Period("2020-01-01")),
            ("regionA", "__total", "__total", pd.Period("2020-01-01")),
            ("regionA", "storeB", "__total", pd.Period("2020-01-01")),
            ("regionA", "storeB", "p1", pd.Period("2020-01-01")),
            ("regionA", "storeB", "p2", pd.Period("2020-01-01")),
            ("regionB", "__total", "__total", pd.Period("2020-01-01")),
            ("regionB", "storeA", "__total", pd.Period("2020-01-01")),
            ("regionB", "storeA", "p1", pd.Period("2020-01-01")),
            ("regionB", "storeA", "p2", pd.Period("2020-01-01")),
        ],
        names=["region", "store", "product", "period"],
    )

    T = 1
    A = 0.9
    AB = 0.8
    AB1 = 0.81
    AB2 = 0.72
    B = 0.7
    BA = 0.63
    BA1 = 0.56
    BA2 = 0.5

    data = [T, A, AB, AB1, AB2, B, BA, BA1, BA2]
    X = pd.Series(data, index=index)

    # Expected outcome:

    data = [
        T,
        A * T,
        AB * A * T,
        AB1 * AB * A * T,
        AB2 * AB * A * T,
        B * T,
        BA * B * T,
        BA1 * BA * B * T,
        BA2 * BA * B * T,
    ]
    expected = pd.Series(data, index=index)
    return X, expected


@pytest.mark.parametrize(
    "example_func", [example_data_for_topdown_all_ones, example_data_for_topdown_varied]
)
def test_recursively_propagate_topdown(example_func):
    """Basic functional test for _recursively_propagate_topdown."""
    X, expected = example_func()  # create the sample data
    X_propagated = _recursively_propagate_topdown(X)

    # Sanity check #1: same shape, same index
    assert X_propagated.shape == X.shape
    pd.testing.assert_index_equal(X_propagated.index, X.index)

    # If all ones, they should remain unchanged
    if (X == 1.0).all():
        pd.testing.assert_series_equal(X, X_propagated)

    pd.testing.assert_series_equal(X_propagated, expected)
    # Otherwise, do a broader check
    # e.g. check that no NaNs introduced, or compare numeric ranges:
    assert not X_propagated.isna().any(), "Should not introduce NaNs."
    assert X_propagated.min() >= 0, "Ratios should stay non-negative (example logic)."

    # If you know the exact numeric outcome for the varied case,
    # you can define a reference 'expected' Series and compare:
    # expected = ...
    # pd.testing.assert_series_equal(X_propagated, expected)
