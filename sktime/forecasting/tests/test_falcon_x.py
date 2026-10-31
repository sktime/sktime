"""Tests for FalconXForecaster update context retention."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.falcon_x import FalconXForecaster


def test_falcon_x_update_restricts_context_to_context_length():
    """``_update`` retains only the latest ``context_length`` observations."""
    y = pd.DataFrame({"a": np.arange(20, dtype=np.float64)})
    forecaster = FalconXForecaster(
        context_length=4, license_accepted=True, endpoint="mock"
    )
    forecaster._fit(y)

    y_new = pd.DataFrame(
        {"a": np.arange(20, 25, dtype=np.float64)}, index=pd.RangeIndex(20, 25)
    )
    forecaster._update(y_new)

    assert len(forecaster.context_) == 4
    np.testing.assert_array_equal(
        forecaster.context_["a"].to_numpy(), [21, 22, 23, 24]
    )


def test_falcon_x_update_no_context_length_keeps_all():
    """With ``context_length=None`` all history is retained in ``context_``."""
    y = pd.DataFrame({"a": np.arange(8, dtype=np.float64)})
    forecaster = FalconXForecaster(
        context_length=None, license_accepted=True, endpoint="mock"
    )
    forecaster._fit(y)

    y_new = pd.DataFrame(
        {"a": np.arange(8, 11, dtype=np.float64)}, index=pd.RangeIndex(8, 11)
    )
    forecaster._update(y_new)

    assert len(forecaster.context_) == 11
