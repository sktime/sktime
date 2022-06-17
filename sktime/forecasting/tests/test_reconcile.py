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

# get all the methods
METHOD_LIST = ReconcilerForecaster.METHOD_LIST


# test the reconciled predictions are actually hierarchical
# test the index/columns on the g and s matrices match
# test it works for named and unnamed indexes
@pytest.mark.parametrize("method", METHOD_LIST)
def test_reconciler_fit_predict(method):
    """Tests fit_predict and output of ReconcilerForecaster.

    Raises
    ------
    This test asserts that the output of ReconcilerForecaster is actually hierarhical
    in that the predictions sum together appropriately. It also tests the index
    and columns of the fitted s and g matrix from each method and finally tests
    if the method works for both named and unnamed indexes
    """
    agg = Aggregator(flatten_single_levels=True)

    y = _bottom_hier_datagen(
        no_bottom_nodes=3,
        no_levels=1,
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
