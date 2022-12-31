#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Tests for hierarchical reconciler forecasters."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen
from sktime.utils.validation._dependencies import _check_soft_dependencies

# get all the methods
METHOD_LIST = ReconcilerForecaster.METHOD_LIST
level_list = [1, 2, 3]
flatten_list = [True, False]


# test the reconciled predictions are actually hierarchical
# test the index/columns on the g and s matrices match
# test it works for named and unnamed indexes
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
        no_bottom_nodes=5,
        no_levels=no_levels,
        random_seed=123,
    )
    # add aggregate levels
    y = agg.fit_transform(y)

    # forecast all levels
    fh = ForecastingHorizon([1, 2], is_relative=True)
    forecaster = ExponentialSmoothing(trend="add", seasonal="additive", sp=12)
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
