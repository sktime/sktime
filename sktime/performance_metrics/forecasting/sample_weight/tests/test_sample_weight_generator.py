"""Tests for sample weight generators."""

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.forecasting.sample_weight._base import (
    BaseSampleWeightGenerator,
)
from sktime.performance_metrics.forecasting.sample_weight._types import (
    check_sample_weight_generator,
)
from sktime.tests.test_switch import run_test_module_changed


class DummyWeightGenerator(BaseSampleWeightGenerator):
    def __call__(self, y_true, y_pred=None, **kwargs) -> None:
        return None


class UniformWeightGenerator(BaseSampleWeightGenerator):
    def __call__(self, y_true, y_pred=None, **kwargs) -> np.ndarray:
        return np.ones_like(y_true)


class RecencyWeightGenerator(BaseSampleWeightGenerator):
    def __init__(self, decay_rate=0.1):
        self.decay_rate = decay_rate

    def __call__(self, y_true, y_pred=None, **kwargs):
        weights = np.exp(-self.decay_rate * np.arange(len(y_true))[::-1])
        return weights / np.sum(weights)


class DateRangeWeightGenerator(BaseSampleWeightGenerator):
    def __init__(self, date_ranges, default_weight=1.0):
        self.date_ranges = date_ranges
        self.default_weight = default_weight

    def _get_dt(self, date_series):
        """Get the datetime object from the series."""
        dt = date_series
        if isinstance(date_series, pd.Timestamp):
            dt = date_series
        elif isinstance(date_series, pd.DatetimeIndex):
            dt = date_series
        elif isinstance(date_series, pd.Series):
            dt = date_series.dt
        elif isinstance(date_series, pd.DataFrame):
            dt = date_series.dt
        elif isinstance(date_series, np.datetime64):
            dt = pd.to_datetime(date_series)
        else:
            msg = f"Invalid type for conversion to datetime: {type(date_series)}"
            raise ValueError(msg)
        return dt

    def __call__(self, y_true, y_pred=None, **kwargs):
        if "dates" not in kwargs:
            msg = "'dates' must be provided in kwargs for DateRangeWeightGenerator"
            raise ValueError(msg)

        if not (
            isinstance(self.date_ranges, list)
            and (
                all(isinstance(range_config, dict) for range_config in self.date_ranges)
            )
        ):
            raise ValueError("'date_ranges' must be a list of dictionaries")

        if not all(
            "start" in range_config
            and "end" in range_config
            and "weight" in range_config
            for range_config in self.date_ranges
        ):
            msg = "'date_ranges' must be a list of dicts with the keys: "
            msg += "'start', 'end', and 'weight'"
            raise ValueError(msg)

        dates = pd.to_datetime(kwargs["dates"])
        weights = np.full(len(y_true), self.default_weight)

        for range_config in self.date_ranges:
            start_date = pd.to_datetime(range_config["start"], format="%m-%d")
            end_date = pd.to_datetime(range_config["end"], format="%m-%d")

            dt = self._get_dt(dates)
            mask = (
                (dt.month >= start_date.month)
                & (dt.month <= end_date.month)
                & (dt.day >= start_date.day)
                & (dt.day <= end_date.day)
            )
            weights[mask] = range_config["weight"]

        return weights / np.sum(weights)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_base_sample_weight_generator():
    """Test BaseSampleWeightGenerator."""

    class TestGenerator(BaseSampleWeightGenerator):
        def __call__(self, y_true, y_pred=None, **kwargs):
            return np.ones_like(y_true)

    generator = TestGenerator()
    y_true = np.array([1, 2, 3])
    weights = generator(y_true)

    assert isinstance(weights, np.ndarray)
    assert np.array_equal(weights, np.ones_like(y_true))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_check_sample_weight_generator():
    """Test check_sample_weight_generator function."""

    def valid_generator(y_true, y_pred=None, **kwargs):
        return np.ones_like(y_true)

    def valid_generator_no_params(y_true, y_pred=None, **kwargs):
        return None

    def valid_generator_no_y_pred(y_true, **kwargs):
        return np.ones_like(y_true)

    assert check_sample_weight_generator(valid_generator)
    assert check_sample_weight_generator(valid_generator_no_params)
    assert check_sample_weight_generator(valid_generator_no_y_pred)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_check_sample_weight_generator_are_invalid():
    """Test check_sample_weight_generator function."""

    def valid_generator_no_kwargs(y_true, y_pred=None):
        return np.ones_like(y_true)

    def invalid_generator():
        return "invalid"

    def invalid_generator_no_param_y_true(invalid):
        return np.ones((1, 1))

    # Validate
    with pytest.raises(ValueError):
        check_sample_weight_generator(valid_generator_no_kwargs)

    with pytest.raises(ValueError):
        check_sample_weight_generator(invalid_generator)

    with pytest.raises(ValueError):
        check_sample_weight_generator(invalid_generator_no_param_y_true)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_dummy_weight_generator():
    generator = DummyWeightGenerator()
    y_true = np.array([1, 2, 3])
    weights = generator(y_true)

    assert weights is None


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_uniform_weight_generator():
    generator = UniformWeightGenerator()
    y_true = np.array([1, 2, 3])
    weights = generator(y_true)

    assert isinstance(weights, np.ndarray)
    assert np.array_equal(weights, np.ones_like(y_true))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_recency_weight_generator():
    generator = RecencyWeightGenerator(decay_rate=0.1)
    y_true = np.array([1, 2, 3])
    weights = generator(y_true)

    assert isinstance(weights, np.ndarray)
    assert len(weights) == len(y_true)
    assert np.all(weights[:-1] < weights[1:])  # Weights should be increasing


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_date_range_weight_generator():
    date_ranges = [
        {"start": "01-01", "end": "03-31", "weight": 2},
        {"start": "06-01", "end": "08-31", "weight": 3},
    ]
    generator = DateRangeWeightGenerator(date_ranges=date_ranges)
    y_true = np.array([1, 2, 3, 4, 5])
    dates = pd.date_range(start="2023-01-01", periods=5, freq="MS")
    weights = generator(y_true, dates=dates)

    assert isinstance(weights, np.ndarray)
    assert len(weights) == len(y_true)
    assert weights[0] > weights[3]  # January should have higher weight than April


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_sample_weight_generator_with_y_pred():
    class TestGenerator(BaseSampleWeightGenerator):
        def __call__(self, y_true, y_pred=None, **kwargs):
            if y_pred is None:
                return np.ones_like(y_true)
            return np.abs(y_true - y_pred)

    generator = TestGenerator()
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.2, 2.8])
    weights = generator(y_true, y_pred)

    assert isinstance(weights, np.ndarray)
    assert np.allclose(weights, np.abs(y_true - y_pred))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_sample_weight_generator_with_kwargs():
    class TestGenerator(BaseSampleWeightGenerator):
        def __call__(self, y_true, y_pred=None, **kwargs):
            factor = kwargs.get("factor", 1)
            return np.ones_like(y_true) * factor

    generator = TestGenerator()
    y_true = np.array([1, 2, 3])
    weights = generator(y_true, factor=2)

    assert isinstance(weights, np.ndarray)
    assert np.array_equal(weights, np.ones_like(y_true) * 2)
