# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for TimerS1Forecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.timer_s1 import TimerS1Forecaster


def test_timer_s1_predict_with_transformers_generation_position_ids():
    """TimerS1 generation uses patch positions, not raw time-step positions."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    forecaster = TimerS1Forecaster(**TimerS1Forecaster.get_test_params()[0])
    y = pd.Series(np.arange(20, dtype=float))

    y_pred = forecaster.fit(y, fh=[1, 2, 3]).predict()

    assert len(y_pred) == 3
