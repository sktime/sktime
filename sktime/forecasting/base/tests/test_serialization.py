# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the zero-shot serialization mixin."""

import pickle

import pytest

from sktime.forecasting.base._serialization import _ZeroShotSerializationMixin


class MockMultitonCachedModel:
    """Mock for an unpicklable model pipeline like a PyTorch module."""
    def __reduce__(self):
        raise TypeError("cannot pickle 'MockMultitonCachedModel' object")


class MockZeroShotForecaster(_ZeroShotSerializationMixin):
    """Mock forecaster implementing the mixin for testing."""

    def __init__(self):
        # Attributes that should be excluded
        self.model_pipeline = MockMultitonCachedModel()
        self._model_pipeline = MockMultitonCachedModel()
        self.predictor_ = MockMultitonCachedModel()
        self.estimator_ = MockMultitonCachedModel()
        self.tfm = MockMultitonCachedModel()
        self.model = MockMultitonCachedModel()
        
        # Attributes that should be preserved
        self.param1 = "preserve_me"
        self.param2 = 42
        self._is_fitted = True


def test_zero_shot_serialization_mixin():
    """Test that the mixin correctly excludes unpicklable attributes."""
    forecaster = MockZeroShotForecaster()
    
    # Assert that the initial setup holds the unpicklable objects
    assert isinstance(forecaster.model_pipeline, MockMultitonCachedModel)
    
    # Perform serialization and deserialization
    try:
        pickled_forecaster = pickle.dumps(forecaster)
        unpickled_forecaster = pickle.loads(pickled_forecaster)
    except TypeError as e:
        pytest.fail(f"Serialization failed: {e}")
    
    # Verify that excluded attributes are set to None
    assert unpickled_forecaster.model_pipeline is None
    assert unpickled_forecaster._model_pipeline is None
    assert unpickled_forecaster.predictor_ is None
    assert unpickled_forecaster.estimator_ is None
    assert unpickled_forecaster.tfm is None
    assert unpickled_forecaster.model is None
    
    # Verify that standard attributes are preserved
    assert unpickled_forecaster.param1 == "preserve_me"
    assert unpickled_forecaster.param2 == 42
    assert unpickled_forecaster._is_fitted is True
