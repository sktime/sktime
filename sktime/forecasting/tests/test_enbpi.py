# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for EnbPIForecaster."""

from sktime.forecasting.enbpi import EnbPIForecaster
from sktime.transformations.bootstrap import (
    MovingBlockBootstrapTransformer,
    TSBootstrapAdapter,
)


class _ExternalBootstrap:
    """Minimal external bootstrap object for constructor tests."""


def test_default_bootstrap_transformer_is_initialized():
    """Test that the default bootstrap transformer is initialized."""
    forecaster = EnbPIForecaster()

    assert isinstance(forecaster.bootstrap_transformer_, MovingBlockBootstrapTransformer)
    assert forecaster.bootstrap_transformer_.return_indices


def test_external_bootstrap_object_is_adapted():
    """Test that external bootstrap objects are wrapped in the tsbootstrap adapter."""
    bootstrap = _ExternalBootstrap()
    forecaster = EnbPIForecaster(bootstrap_transformer=bootstrap)

    assert isinstance(forecaster.bootstrap_transformer_, TSBootstrapAdapter)
    assert forecaster.bootstrap_transformer_.bootstrap is bootstrap
    assert forecaster.bootstrap_transformer_.return_indices
