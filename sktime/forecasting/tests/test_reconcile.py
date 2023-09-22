#!/usr/bin/env python3 -u
"""Tests for hierarchical reconciler forecasters."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen, _make_hierarchical
from sktime.utils.validation._dependencies import _check_soft_dependencies

# get all the methods
METHOD_LIST = ReconcilerForecaster.METHOD_LIST
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
    This test asserts that the output of ReconcilerForecaster is actually hierarhical
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

    # check the row index and column indexes match
    msg = "Summation index/columns and G matrix index/columns do not match."
    assert np.all(reconciler.g_matrix.columns == reconciler.s_matrix.index), msg
    assert np.all(reconciler.g_matrix.index == reconciler.s_matrix.columns), msg

    # check if we now remove aggregate levels and use Aggregator it is equal
    prds_recon_bottomlevel = agg.inverse_transform(prds_recon)
    assert_frame_equal(prds_recon, agg.fit_transform(prds_recon_bottomlevel))

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
