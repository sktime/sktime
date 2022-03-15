# -*- coding: utf-8 -*-
"""Tests for Mock estimators base classes and utils."""

import pytest

from sktime.classification.base import BaseClassifier
from sktime.clustering.base import BaseClusterer
from sktime.forecasting.base import BaseForecaster
from sktime.transformations.base import BaseTransformer
from sktime.utils.estimators._base import _method_logger, _MockEstimatorMixin


@pytest.mark.parametrize(
    "base", [BaseForecaster, BaseClassifier, BaseClusterer, BaseTransformer]
)
def test_mixin(base):
    """Test _MockEstimatorMixin is valid for all sktime estimator base classes."""

    class _DummyClass(base, _MockEstimatorMixin):
        def __init__(self):
            super(_DummyClass, self).__init__()

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
    assert hasattr(dummy_instance, "_log") & hasattr(dummy_instance, "log")


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
            super(_DummyClass, self).__init__()

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
