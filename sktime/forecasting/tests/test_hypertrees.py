"""Tests for the hypertrees forecasters."""

import pandas as pd
import pytest

from sktime.forecasting.hypertrees import (
    HyperTreeARForecaster,
    HyperTreeNetARForecaster,
)
from sktime.tests.test_switch import run_test_for_class

FORECASTERS = [HyperTreeARForecaster, HyperTreeNetARForecaster]


@pytest.mark.skipif(
    not run_test_for_class(FORECASTERS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("forecaster_class", FORECASTERS)
@pytest.mark.parametrize("period_freq", ["M", "Q", "Y", "D"])
def test_period_index_freq_is_offset_alias(forecaster_class, period_freq):
    """Test that a PeriodIndex frequency is coerced to a valid DateOffset alias.

    Failure condition of #11228.

    The forecasters build an internal synthetic ``pd.date_range`` from the
    frequency of ``y``. Reading it off ``index.freqstr`` yields a period alias,
    e.g., ``"M"`` for a monthly ``PeriodIndex``, which ``pd.date_range`` rejects
    from pandas 3 on with ``ValueError: Invalid frequency: M``.
    """
    y = pd.Series(
        range(30),
        index=pd.period_range("2000-01-01", periods=30, freq=period_freq),
        dtype="float64",
    )
    params = forecaster_class.get_test_params()[0]
    forecaster = forecaster_class(**params)
    forecaster.fit(y, fh=[1, 2, 3])

    # the resolved frequency must be parseable as a DateOffset, on any pandas
    pd.tseries.frequencies.to_offset(forecaster._freq)

    y_pred = forecaster.predict()
    assert list(y_pred.index) == [y.index[-1] + i for i in (1, 2, 3)]
