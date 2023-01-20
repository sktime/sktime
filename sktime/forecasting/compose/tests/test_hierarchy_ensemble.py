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
        [("ptf", PolynomialTrendForecaster(), 0)],
        [("naive", NaiveForecaster(), 0)],
    ],
)
def test_hierarchy_ensemble_level_predict(forecasters):
    """Check the level predictions."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=3,
        no_levels=1,
        random_seed=123,
    )

    forecaster_withdef = HierarchyEnsembleForecaster(
        forecasters, default=forecasters[0][1].clone()
    )

    forecaster_withdef.fit(y, fh=[1, 2, 3])
    actual_withdef = forecaster_withdef.predict()

    y = agg.fit_transform(y)

    test_forecaster = forecasters[0][1].clone()
    test_forecaster.fit(y, fh=[1, 2, 3])
    test_withdef = test_forecaster.predict()

    msg = "Level predictions do not match"
    assert np.all(actual_withdef == test_withdef), msg


@pytest.mark.parametrize(
    "forecasters",
    [
        [("ptf", PolynomialTrendForecaster(), [("__total"), ("l1_node01")])],
        [("naive", NaiveForecaster(), [("__total"), ("l1_node02")])],
    ],
)
def test_hierarchy_ensemble_node_predict(forecasters):
    """Check the node predictions."""
    agg = Aggregator()

    y = _bottom_hier_datagen(
        no_bottom_nodes=3,
        no_levels=1,
        random_seed=123,
    )

    forecaster_withdef = HierarchyEnsembleForecaster(
        forecasters, by="node", default=forecasters[0][1].clone()
    )

    forecaster_withdef.fit(y, fh=[1, 2, 3])
    actual_withdef = forecaster_withdef.predict()

    y = agg.fit_transform(y)

    test_forecaster = forecasters[0][1].clone()
    test_forecaster.fit(y, fh=[1, 2, 3])
    test_withdef = test_forecaster.predict()

    msg = "Node predictions do not match"
    assert np.all(actual_withdef == test_withdef), msg
