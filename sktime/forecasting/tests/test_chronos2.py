"""Tests for Chronos2Forecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.chronos2 import Chronos2Forecaster
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(Chronos2Forecaster, severity="none"),
    reason="autots not available",
)
def test_chronos2_fit_truncates_context_on_time_axis():
    """`context_length` truncation should apply to time axis, not feature axis."""
    pytest.importorskip("torch")

    y = pd.DataFrame(np.arange(300).reshape(100, 3), columns=["a", "b", "c"])
    forecaster = Chronos2Forecaster(config={"context_length": 10}, ignore_deps=True)
    forecaster._load_pipeline = lambda: object()

    forecaster._fit(y)

    assert forecaster._context.shape == (3, 10)
