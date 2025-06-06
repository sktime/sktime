import functools

import pandas as pd
import pytest

from sktime.transformations.hierarchical.reconcile._utils import (
    _get_series_for_each_hierarchical_level,
)


@pytest.mark.parametrize(
    "hierarchical_level_nodes",
    [
        (
            pd.MultiIndex.from_tuples([("__total", "__total")]),
            pd.MultiIndex.from_tuples([("l1", "__total"), ("l2", "__total")]),
            pd.MultiIndex.from_tuples(
                [("l1", "l1_2"), ("l1", "l1_1"), ("l2", "l2_1"), ("l2", "l2_4")]
            ),
        ),
        (
            pd.Index(["__total"]),
            pd.Index(["l1", "l2"]),
        ),
        (
            pd.MultiIndex.from_tuples([("__total", "__total", "__total")]),
            pd.MultiIndex.from_tuples([("l1", "__total", "__total")]),
            pd.MultiIndex.from_tuples(
                [("l1", "l1_2", "__total"), ("l1", "l1_1", "__total")]
            ),
            pd.MultiIndex.from_tuples(
                [
                    ("l1", "l1_2", "l1_2_1"),
                    ("l1", "l1_2", "l1_2_2"),
                    ("l1", "l1_1", "l1_1_1"),
                    ("l1", "l1_1", "l1_1_2"),
                ]
            ),
        ),
    ],
)
def test_get_series_for_each_hierarchical_level(hierarchical_level_nodes):
    full_index = functools.reduce(lambda a, b: a.union(b), hierarchical_level_nodes)
    out = _get_series_for_each_hierarchical_level(full_index)

    for i, level in enumerate(hierarchical_level_nodes):
        assert set(level) == set(out[i])
