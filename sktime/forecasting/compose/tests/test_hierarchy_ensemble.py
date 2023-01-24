#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of HierarchyEnsembleForecaster functionality."""

__author__ = ["VyomkeshVyas"]

import numpy as np
import pytest

from sktime.forecasting.compose import HierarchyEnsembleForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependencies not available",
)
@pytest.mark.parametrize(
    "forecasters",
    [
        [("ptf", PolynomialTrendForecaster(), 0), ("naive", NaiveForecaster(), 1)],
        [("naive", NaiveForecaster(), 0)],
    ],
)
def test_hierarchy_ensemble_level_predict(forecasters):
    """Check the level predictions."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=7,
        no_levels=2,
        random_seed=123,
    )

    forecaster = HierarchyEnsembleForecaster(
        forecasters, default=forecasters[0][1].clone()
    )

    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    y = agg.fit_transform(y)

    for i in range(len(forecasters)):
        test_frcstr = forecasters[i][1].clone()
        test_frcstr.fit(y.loc[forecaster.fitted_list[i][1]], fh=[1, 2, 3])
        test_pred = test_frcstr.predict()
        msg = "Level predictions do not match"
        assert np.all(actual_pred.loc[test_pred.index] == test_pred), msg

    def_frcstr = forecasters[0][1].clone()
    def_frcstr.fit(y.loc[forecaster.fitted_list[-1][1]], fh=[1, 2, 3])
    def_pred = def_frcstr.predict()
    msg = "Level default predictions do not match"
    assert np.all(actual_pred.loc[def_pred.index] == def_pred), msg


@pytest.mark.parametrize(
    "forecasters",
    [
        [
            ("ptf", PolynomialTrendForecaster(), [("__total", "__total")]),
            ("naive", NaiveForecaster(), [("l2_node01", "__total")]),
        ],
        [("naive", NaiveForecaster(), [("l2_node01", "l1_node06")])],
    ],
)
def test_hierarchy_ensemble_node_predict(forecasters):
    """Check the node predictions."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=7,
        no_levels=2,
        random_seed=123,
    )

    forecaster = HierarchyEnsembleForecaster(
        forecasters, by="node", default=forecasters[0][1].clone()
    )

    forecaster.fit(y, fh=[1, 2, 3])
    actual_pred = forecaster.predict()

    y = agg.fit_transform(y)

    for i in range(len(forecasters)):
        test_frcstr = forecasters[i][1].clone()
        test_frcstr.fit(y.loc[forecaster.fitted_list[i][1]], fh=[1, 2, 3])
        test_pred = test_frcstr.predict()
        msg = "Node predictions do not match"
        assert np.all(actual_pred.loc[test_pred.index] == test_pred), msg

    def_frcstr = forecasters[0][1].clone()
    def_frcstr.fit(y.loc[forecaster.fitted_list[-1][1]], fh=[1, 2, 3])
    def_pred = def_frcstr.predict()
    msg = "Node default predictions do not match"
    assert np.all(actual_pred.loc[def_pred.index] == def_pred), msg
