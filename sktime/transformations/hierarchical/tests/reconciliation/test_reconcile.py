#!/usr/bin/env python3 -u
"""Tests for hierarchical reconcilers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconciliation import (
    BottomUpReconciler,
    ForecastProportions,
    MiddleOutReconciler,
    TopdownShareReconciler,
)
from sktime.transformations.hierarchical.reconciliation._utils import (
    _get_series_for_each_hierarchical_level,
    loc_series_idxs,
)
from sktime.utils._testing.hierarchical import _bottom_hier_datagen


def _generate_unreconciled_hierarchical_data(
    flatten_single_levels, no_levels, no_bottom_nodes=5
):
    agg = Aggregator(flatten_single_levels=flatten_single_levels)

    X = _bottom_hier_datagen(
        no_bottom_nodes=no_bottom_nodes,
        no_levels=no_levels,
        random_seed=123,
    )
    # add aggregate levels
    X = agg.fit_transform(X)

    prds = X
    return prds


@pytest.mark.parametrize(
    "reconciler, expected_immutable_level",
    [
        (BottomUpReconciler, -1),
        (TopdownShareReconciler, 0),
        (ForecastProportions, 0),
        (MiddleOutReconciler, "middle_level"),
    ],
)
@pytest.mark.parametrize("flatten_single_levels", [True, False])
@pytest.mark.parametrize("no_levels", [2, 3, 4])
def test_reconcilers_keep_immutable_levels(
    reconciler, expected_immutable_level, flatten_single_levels, no_levels
):
    y = _generate_unreconciled_hierarchical_data(
        flatten_single_levels=flatten_single_levels, no_levels=no_levels
    )

    test_params = reconciler.get_test_params()
    if not test_params:
        test_params = [{}]

    for test_param in test_params:
        instance = reconciler(**test_param)

        instance.fit(y)
        yt = instance.transform(y)
        yt = yt + np.random.normal(0, 10, (yt.shape[0], 1))
        y_reconc = instance.inverse_transform(yt)

        hierarchical_level_nodes = _get_series_for_each_hierarchical_level(
            y.index.droplevel(-1).unique()
        )

        _expected_immutable_level = expected_immutable_level
        if isinstance(expected_immutable_level, str):
            _expected_immutable_level = getattr(instance, expected_immutable_level)

        immutable_level_series = hierarchical_level_nodes[_expected_immutable_level]
        y_immutable_reconc = loc_series_idxs(y_reconc, immutable_level_series)
        y_immutable = loc_series_idxs(yt, immutable_level_series)
        assert_frame_equal(y_immutable_reconc, y_immutable)
        # Assert that other levels have changed
        assert not y_reconc.equals(y)
