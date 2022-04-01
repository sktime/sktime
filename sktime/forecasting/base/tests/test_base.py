# -*- coding: utf-8 -*-
"""Testing advanced functionality of the base class."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from functools import reduce
from operator import mul

import pytest

from sktime.datatypes import check_is_mtype, convert, get_examples
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.base import BaseForecaster
from sktime.utils._testing.deep_equals import deep_equals
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_panel_X

PANEL_MTYPES = ["pd-multiindex", "nested_univ", "numpy3D"]
HIER_MTYPES = ["pd_multiindex_hier"]


@pytest.mark.parametrize("mtype", PANEL_MTYPES)
def test_vectorization_series_to_panel(mtype):
    """Test that forecaster vectorization works for Panel data.

    This test passes Panel data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    n_instances = 10

    y = _make_panel_X(n_instances=n_instances, random_state=42)
    y = convert(y, from_type="nested_univ", to_type=mtype)

    y_pred = ARIMA().fit(y).predict([1, 2, 3])
    valid, _, metadata = check_is_mtype(y_pred, mtype, return_metadata=True)

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )

    assert valid, msg

    y_pred_instances = metadata["n_instances"]
    msg = (
        f"vectorization test produces wrong number of instances "
        f"expected {n_instances}, found {y_pred_instances}"
    )

    assert y_pred_instances == n_instances, msg

    y_pred_equal_length = metadata["is_equal_length"]
    msg = (
        "vectorization test produces non-equal length Panel forecast, should be "
        "equal length, and length equal to the forecasting horizon [1, 2, 3]"
    )
    assert y_pred_equal_length, msg


@pytest.mark.parametrize("mtype", HIER_MTYPES)
def test_vectorization_series_to_hier(mtype):
    """Test that forecaster vectorization works for Panel data.

    This test passes Panel data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    hierarchy_levels = (2, 4)
    n_instances = reduce(mul, hierarchy_levels)

    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)
    y = convert(y, from_type="pd_multiindex_hier", to_type=mtype)

    y_pred = ARIMA().fit(y).predict([1, 2, 3])
    valid, _, metadata = check_is_mtype(y_pred, mtype, return_metadata=True)

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )

    assert valid, msg

    y_pred_instances = metadata["n_instances"]
    msg = (
        f"vectorization test produces wrong number of instances "
        f"expected {n_instances}, found {y_pred_instances}"
    )

    assert y_pred_instances == n_instances, msg

    y_pred_equal_length = metadata["is_equal_length"]
    msg = (
        "vectorization test produces non-equal length Panel forecast, should be "
        "equal length, and length equal to the forecasting horizon [1, 2, 3]"
    )
    assert y_pred_equal_length, msg


class MockPredInt(BaseForecaster):
    """Mock class to override _predict_interval with an example return."""

    def __init__(self, example_number):
        self.example_number = example_number

    def _predict_interval(self, fh, X=None, coverage=0.9):
        ret = get_examples("pred_interval", "Proba")[self.example_number]
        idx = ret.columns.get_level_values(1).isin(coverage)
        return ret.iloc[:, idx]


class MockPredQuantiles(BaseForecaster):
    """Mock class to override _predict_quantiles with an example return."""

    def __init__(self, example_number):
        self.example_number = example_number

    def _predict_quantiles(self, fh, X=None, alpha=0.9):
        ret = get_examples("pred_quantiles", "Proba")[self.example_number]
        idx = ret.columns.get_level_values(1).isin(alpha)
        return ret.iloc[:, idx]


# indices of examples which have lower/upper pairs of quantiles
EXAMPLE_INDS = [2, 3]


@pytest.mark.parametrize("example_number", EXAMPLE_INDS)
def test_base_interval_to_quantiles(example_number):
    """Test functionality to convert interval to quantile forecasts in base class."""
    y_pred_expected = get_examples("pred_quantiles", "Proba")[example_number]
    alpha = y_pred_expected.columns.get_level_values(1).unique()

    est = MockPredInt(example_number)
    y_pred = est._predict_quantiles(0, 0, alpha=alpha)

    assert deep_equals(y_pred, y_pred_expected)


@pytest.mark.parametrize("example_number", EXAMPLE_INDS)
def test_base_quantiles_to_interval(example_number):
    """Test functionality to convert quantile to intervalforecasts in base class."""
    y_pred_expected = get_examples("pred_interval", "Proba")[example_number]
    coverage = y_pred_expected.columns.get_level_values(1).unique()

    est = MockPredQuantiles(example_number)
    y_pred = est._predict_interval(0, 0, coverage=coverage)

    assert deep_equals(y_pred, y_pred_expected)
