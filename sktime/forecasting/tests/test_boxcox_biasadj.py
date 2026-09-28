"""Tests for BoxCoxBiasAdjustedForecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.boxcox_biasadj import BoxCoxBiasAdjustedForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(BoxCoxBiasAdjustedForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_boxcox_biasadj_fallback_on_bracket_error():
    """Regression test for #10301.

    ``BoxCoxTransformer`` can fail to find a valid optimization bracket for the
    Box-Cox lambda parameter on some inputs (e.g. series starting at zero),
    raising ``scipy.optimize.BracketError``, a ``RuntimeError`` subclass.
    ``BoxCoxBiasAdjustedForecaster._fit`` should catch this and fall back to
    the identity transform (``lambda=1.0``) instead of letting the fit fail.
    """
    y = pd.Series(
        range(20), index=pd.date_range("2020-01-01", periods=20, freq="D"), name="y"
    )
    forecaster = BoxCoxBiasAdjustedForecaster(
        forecaster=NaiveForecaster(strategy="mean")
    )

    with pytest.warns(RuntimeWarning, match="Falling back to lambda=1.0"):
        forecaster.fit(y, fh=[1])

    assert forecaster.boxcox_transformer_.lambda_ == 1.0

    y_pred = forecaster.predict()
    assert np.isfinite(y_pred).all()
