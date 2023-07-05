"""Tests for Mock estimators base classes and utils."""

__author__ = ["ltsaprounis"]

import pytest
from pandas.testing import assert_series_equal

from sktime.classification.base import BaseClassifier
from sktime.clustering.base import BaseClusterer
from sktime.datasets import load_airline
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.utils.estimators import make_mock_estimator
from sktime.utils.estimators._base import _method_logger, _MockEstimatorMixin

y_series = load_airline().iloc[:-5]


@pytest.mark.parametrize(
    "base", [BaseForecaster, BaseClassifier, BaseClusterer, BaseTransformer]
)
def test_mixin(base):
    """Test _MockEstimatorMixin is valid for all sktime estimator base classes."""

    class _DummyClass(base, _MockEstimatorMixin):
        def __init__(self):
            super().__init__()

        def _fit(self):
            """Empty method, here for testing purposes."""
            pass

        def _predict(self):
            """Empty method, here for testing purposes."""
            pass

        def _score(self):
            """Empty method, here for testing purposes."""
            pass

    dummy_instance = _DummyClass()
    assert hasattr(dummy_instance, "log")
    dummy_instance.add_log_item(42)
    assert hasattr(dummy_instance, "_MockEstimatorMixin__log")


def test_add_log_item():
    """Test _MockEstimatorMixin.add_log_item behaviour."""
    mixin = _MockEstimatorMixin()
    mixin.add_log_item(1)
    mixin.add_log_item(2)
    assert mixin.log[0] == 1
    assert mixin.log[1] == 2


def test_log_is_property():
    """Test _MockEstimatorMixin.log can't be overwritten."""
    mixin = _MockEstimatorMixin()
    with pytest.raises(AttributeError) as excinfo:
        mixin.log = 1
        assert "can't set attribute" in str(excinfo.value)


def test_method_logger_exception():
    """Test that _method_logger only works for _MockEstimatorMixin subclasses."""

    class _DummyClass:
        def __init__(self) -> None:
            """Empty method, here for testing purposes."""
            pass

        @_method_logger
        def _method(self):
            """Empty method, here for testing purposes."""
            pass

    with pytest.raises(TypeError) as excinfo:
        dummy_instance = _DummyClass()
        dummy_instance._method()
        assert "Estimator is not a Mock Estimator" in str(excinfo.value)


def test_method_logger():
    """Test that method logger returns the correct output."""

    class _DummyClass(_MockEstimatorMixin):
        def __init__(self) -> None:
            super().__init__()

        @_method_logger
        def _method1(self, positional_param, optional_param="test_optional"):
            """Empty method, here for testing purposes."""
            pass

        @_method_logger
        def _method2(self, positional_param, optional_param="test_optional_2"):
            """Empty method, here for testing purposes."""
            pass

        @_method_logger
        def _method3(self):
            """Empty method, here for testing purposes."""
            pass

    dummy_instance = _DummyClass()
    dummy_instance._method1("test_positional")
    assert dummy_instance.log == [
        (
            "_method1",
            {"positional_param": "test_positional", "optional_param": "test_optional"},
        )
    ]
    dummy_instance._method2("test_positional_2")
    assert dummy_instance.log == [
        (
            "_method1",
            {"positional_param": "test_positional", "optional_param": "test_optional"},
        ),
        (
            "_method2",
            {
                "positional_param": "test_positional_2",
                "optional_param": "test_optional_2",
            },
        ),
    ]
    dummy_instance._method3()
    assert dummy_instance.log == [
        (
            "_method1",
            {"positional_param": "test_positional", "optional_param": "test_optional"},
        ),
        (
            "_method2",
            {
                "positional_param": "test_positional_2",
                "optional_param": "test_optional_2",
            },
        ),
        ("_method3", {}),
    ]


@pytest.mark.parametrize(
    "estimator_class, method_regex, logged_methods",
    [
        (NaiveForecaster, r"(?!^_\w+)", ["fit"]),
        (NaiveForecaster, ".*", ["fit", "_fit"]),
        (BoxCoxTransformer, r"(?!^_\w+)", ["fit"]),
        (BoxCoxTransformer, ".*", ["fit", "_fit"]),
    ],
)
def test_make_mock_estimator(estimator_class, method_regex, logged_methods):
    """Test that make_mock_estimator output logs the right methods."""
    estimator = make_mock_estimator(estimator_class, method_regex)()
    estimator.fit(y_series)
    methods_called = [entry[0] for entry in estimator.log]

    assert set(methods_called) >= set(logged_methods)


@pytest.mark.parametrize(
    "estimator_class, estimator_kwargs",
    [
        (NaiveForecaster, {"strategy": "last", "sp": 2, "window_length": None}),
        (NaiveForecaster, {"strategy": "mean", "sp": 1, "window_length": None}),
    ],
)
def test_make_mock_estimator_with_kwargs(estimator_class, estimator_kwargs):
    """Test that make_mock_estimator behaves like the passed estimator."""
    mock_estimator = make_mock_estimator(estimator_class)
    mock_estimator_instance = mock_estimator(estimator_kwargs)
    estimator_instance = estimator_class(**estimator_kwargs)
    mock_estimator_instance.fit(y_series)
    estimator_instance.fit(y_series)

    assert_series_equal(
        estimator_instance.predict(fh=[1, 2, 3]),
        mock_estimator_instance.predict(fh=[1, 2, 3]),
    )
    assert (
        (mock_estimator_instance.strategy == estimator_kwargs["strategy"])
        and (mock_estimator_instance.sp == estimator_kwargs["sp"])
        and (mock_estimator_instance.window_length == estimator_kwargs["window_length"])
    )
