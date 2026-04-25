#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for EnbPIForecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.enbpi import EnbPIForecaster
from sktime.libs.tsbootstrap import BlockBootstrap
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(EnbPIForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_enbpi_default_bootstrap_transformer_initializes_and_predicts():
    """Default constructor should initialize a valid bootstrap transformer."""
    y = load_airline().to_frame()
    fh = ForecastingHorizon(np.arange(1, 4), is_relative=True)

    forecaster = EnbPIForecaster()
    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict(fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) == len(fh)


@pytest.mark.skipif(
    not run_test_for_class(EnbPIForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_enbpi_with_vendored_blockbootstrap_predicts_interval():
    """Vendored BlockBootstrap should be accepted through TSBootstrapAdapter."""
    y = load_airline().to_frame()
    fh = ForecastingHorizon(np.arange(1, 4), is_relative=True)

    forecaster = EnbPIForecaster(
        bootstrap_transformer=BlockBootstrap(n_bootstraps=3, block_length=4)
    )
    forecaster.fit(y, fh=fh)
    pred_int = forecaster.predict_interval(fh=fh, coverage=[0.8])

    assert isinstance(pred_int, pd.DataFrame)
    assert len(pred_int) == len(fh)
