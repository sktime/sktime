#!/usr/bin/env python3 -u
"""Tests for hierarchical reconcilers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile import (
    BottomUpReconciler,
    MiddleOutReconciler,
    OptimalReconciler,
    TopdownReconciler,
)
from sktime.transformations.hierarchical.reconcile._optimal import (
    _create_summing_matrix_from_index,
)
from sktime.transformations.hierarchical.reconcile._utils import (
    _get_series_for_each_hierarchical_level,
    _loc_series_idxs,
)
from sktime.utils._testing.hierarchical import _bottom_hier_datagen


def _generate_hier_data(flatten_single_levels, no_levels, no_bottom_nodes=5):
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
        (TopdownReconciler, 0),
        (MiddleOutReconciler, "middle_level"),
    ],
)
@pytest.mark.parametrize("flatten_single_levels", [True, False])
@pytest.mark.parametrize("no_levels", [2, 3, 4])
def test_reconcilers_keep_immutable_levels(
    reconciler, expected_immutable_level, flatten_single_levels, no_levels
):
    y = _generate_hier_data(
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

        # Some reconcilers may change the index levels, by dropping
        # redundant levels, so we need to adjust the immutable level
        levels_droped_at_transform = list(
            set(immutable_level_series.names).difference(yt.index.names)
        )
        immutable_level_series_transformed = immutable_level_series.droplevel(
            levels_droped_at_transform
        )
        y_immutable = _loc_series_idxs(yt, immutable_level_series_transformed)

        y_immutable_reconc = _loc_series_idxs(y_reconc, immutable_level_series)
        y_immutable_reconc = y_immutable_reconc.droplevel(levels_droped_at_transform)

        assert y_immutable_reconc.shape[0] > 0
        assert_frame_equal(y_immutable_reconc, y_immutable)
        # Assert that other levels have changed
        assert not y_reconc.equals(y)


def test_optimal_reconciliation_ols():
    """Test optimal reconciliation with OLS.

    Check if the vector from original forecasts to reconciled forecasts
    is orthogonal to the reconciliation plane.
    """

    y = _generate_hier_data(flatten_single_levels=True, no_levels=3)
    # Keep only one timepoint
    y = y.loc[y.index.get_level_values(-1) == y.index.get_level_values(-1).max()]
    reconciler = OptimalReconciler("ols")
    reconciler.fit(y)
    yt = reconciler.transform(y)
    yt = yt + np.random.normal(0, 10, (yt.shape[0], 1))
    y_reconc = reconciler.inverse_transform(yt)

    # We have to assert that the angle between the reconciled series
    # and the reconciliation plane is orthogonal
    diff = y_reconc - yt
    S = _create_summing_matrix_from_index(y.index.droplevel(-1).unique())
    projection = S.T @ diff.droplevel(-1)
    assert np.allclose(projection, 0, atol=1e-10)
