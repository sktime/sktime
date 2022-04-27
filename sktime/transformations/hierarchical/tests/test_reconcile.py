#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Tests for hierarchical reconcilers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ciaran-g"]

import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile import Reconciler
from sktime.utils._testing.hierarchical import _bottom_hier_datagen

# get all the methods
METHOD_LIST = Reconciler.METHOD_LIST


# test the reconciled predictions are actually hierarchical
@pytest.mark.parametrize("method", METHOD_LIST)
def test_reconciler_fit_transform(method):
    """Tests fit_trasnform and output of reconciler.

    Raises
    ------
    This test asserts that the output of Reconciler is actually hierarhical
    in that the predictions sum together appropriately.
    """
    agg = Aggregator(flatten_single_levels=True)

    X = _bottom_hier_datagen(
        no_bottom_nodes=3,
        no_levels=1,
    )
    # add aggregate levels
    X = agg.fit_transform(X)

    # forecast all levels
    fh = ForecastingHorizon([1, 2], is_relative=True)
    forecaster = ExponentialSmoothing(trend="add", seasonal="additive", sp=12)
    prds = forecaster.fit(X).predict(fh)

    # reconcile forecasts
    reconciler = Reconciler(method=method)
    prds_recon = reconciler.fit_transform(prds)

    # check if we now remove aggregate levels and use Aggregator it is equal
    prds_recon_bottomlevel = agg.inverse_transform(prds_recon)
    msg = "Reonciler returns predictions which do not sum appropriately."

    assert prds_recon.equals(agg.fit_transform(prds_recon_bottomlevel)), msg


# from sktime.utils.estimator_checks import check_estimator
# check_estimator(Reconciler)
