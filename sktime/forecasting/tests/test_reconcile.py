#!/usr/bin/env python3 -u
"""Tests for hierarchical reconciler forecasters."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import (
    _bottom_hier_datagen,
    _make_hierarchical,
    _make_index,
)
from sktime.utils.dependencies import _check_soft_dependencies

# get all the methods
METHOD_LIST = [x for x in ReconcilerForecaster.METHOD_LIST if not x.endswith("nonneg")]
level_list = [1, 2, 3]
flatten_list = [True, False]


# test the reconciled predictions are actually hierarchical
# test the index/columns on the g and s matrices match
# test it works for named and unnamed indexes
@pytest.mark.skipif(
    not run_test_for_class(ReconcilerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("method", METHOD_LIST)
@pytest.mark.parametrize("flatten", flatten_list)
@pytest.mark.parametrize("no_levels", level_list)
def test_reconciler_fit_predict(method, flatten, no_levels):
    """Tests fit_predict and output of ReconcilerForecaster.

    Raises
    ------
    This test asserts that the output of ReconcilerForecaster is actually hierarchical
    in that the predictions sum together appropriately. It also tests the index
    and columns of the fitted s and g matrix from each method and finally tests
    if the method works for both named and unnamed indexes
    """
    agg = Aggregator(flatten_single_levels=flatten)

    y = _bottom_hier_datagen(
        no_bottom_nodes=4,
        no_levels=no_levels,
        random_seed=123,
        length=10,
    )
    # add aggregate levels
    y = agg.fit_transform(y)

    # forecast all levels
    fh = ForecastingHorizon([1, 2], is_relative=True)
    forecaster = ExponentialSmoothing(trend="add", seasonal="additive", sp=3)
    reconciler = ReconcilerForecaster(forecaster, method=method)
    reconciler.fit(y)
    prds_recon = reconciler.predict(fh=fh)

    # Aggregate to check if
    prds_recon_bu = agg.transform(agg.inverse_transform(prds_recon))
    assert_frame_equal(prds_recon, prds_recon_bu)

    # check with unnamed indexes
    y.index.rename([None] * y.index.nlevels, inplace=True)
    reconciler_unnamed = ReconcilerForecaster(forecaster, method=method)
    msg = "Reconciler returns different output for named and unnamed indexes."
    assert prds_recon.equals(reconciler_unnamed.fit_predict(y=y, fh=fh)), msg


@pytest.mark.skipif(
    not run_test_for_class(ReconcilerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("n_columns", [1, 2])
def test_reconcilerforecaster_exog(n_columns):
    """Test that ReconcilerForecaster works without aggregated input, see #3980."""
    from sktime.datatypes._utilities import get_window
    from sktime.forecasting.reconcile import ReconcilerForecaster
    from sktime.forecasting.sarimax import SARIMAX

    y = _make_hierarchical(
        hierarchy_levels=(2, 3),
        n_columns=n_columns,
        min_timepoints=12,
        max_timepoints=12,
        index_type="period",
    )
    y_train = get_window(y, lag=2)
    y_test = get_window(y, window_length=2)

    X = _make_hierarchical(
        hierarchy_levels=(2, 3),
        n_columns=2,
        min_timepoints=12,
        max_timepoints=12,
        index_type="period",
    )
    X.columns = ["foo", "bar"]
    X_train = get_window(X, lag=2)
    X_test = get_window(X, window_length=2)

    forecaster = SARIMAX()
    estimator_instance = ReconcilerForecaster(forecaster, method="mint_shrink")
    fh = [1, 2]
    estimator_instance.fit(y=y_train, X=X_train, fh=fh)
    estimator_instance.predict(X=X_test)
    estimator_instance.update(y=y_test, X=X_test)


@pytest.mark.skipif(
    not run_test_for_class(ReconcilerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", METHOD_LIST)
@pytest.mark.parametrize("return_totals", [True, False])
def test_reconcilerforecaster_return_totals(method, return_totals):
    """Test that ReconcilerForecaster returns the dataframe without the dunder levels"""
    from sktime.datatypes._utilities import get_window
    from sktime.forecasting.compose import YfromX
    from sktime.forecasting.reconcile import ReconcilerForecaster

    m = 2
    n = 2

    y = _make_hierarchical(
        hierarchy_levels=(m, n),
        n_columns=1,
        min_timepoints=12,
        max_timepoints=12,
        index_type="period",
    )
    y_train = get_window(y, lag=2)
    y_test = get_window(y, window_length=2)

    X = _make_hierarchical(
        hierarchy_levels=(m, n),
        n_columns=2,
        min_timepoints=12,
        max_timepoints=12,
        index_type="period",
    )
    X.columns = ["foo", "bar"]
    X_train = get_window(X, lag=2)
    X_test = get_window(X, window_length=2)

    forecaster = YfromX.create_test_instance()
    estimator_instance = ReconcilerForecaster(
        forecaster, method=method, return_totals=return_totals
    )
    fh = [1, 2]
    estimator_instance.fit(y=y_train, X=X_train, fh=fh)
    y_pred = estimator_instance.predict(X=X_test)
    if return_totals:
        # for hierarchy_levels=(m, n), len(y_pred) = len(y_test) + (1 + m) * 2
        assert len(y_pred) == (len(y_test) + (1 + m) * 2)
    else:
        assert len(y_test) == len(y_pred)
        assert y_test.index.equals(y_pred.index)


@pytest.mark.parametrize("alpha", [0, 1])
def test_reconcilerforecaster_singular(alpha):
    """
    Test that ReconcilerForecaster handles highly correlated series,
    where alpha=0 leads to a singular residual covariance matrix.
    """
    hierarchy_levels = (1, 2)
    levels = [
        [f"h{i}_{j}" for j in range(hierarchy_levels[i])]
        for i in range(len(hierarchy_levels))
    ]
    level_names = [f"h{i}" for i in range(len(hierarchy_levels))]

    time_index = _make_index(12, None)
    index = pd.MultiIndex.from_product(
        levels + [time_index], names=level_names + ["time"]
    )
    input_data = np.array([i for i in range(24)])
    input_data = np.stack([input_data, input_data], 1)
    n_columns = 2

    df = pd.DataFrame(
        data=input_data, index=index, columns=[f"c{i}" for i in range(n_columns)]
    )
    base_forecaster = NaiveForecaster(strategy="mean")

    forecaster = ReconcilerForecaster(
        forecaster=base_forecaster, method="mint_cov", alpha=alpha
    )
    if alpha == 0:
        with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
            forecaster.fit(df, fh=[1]).predict()
