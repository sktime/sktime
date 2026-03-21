"""Tests for Moirai2Forecaster."""

__author__ = ["priyanshuharshbodhi1"]

import pandas as pd
import pytest

from sktime.forecasting.moirai2_forecaster import Moirai2Forecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical


@pytest.mark.skipif(
    not run_test_for_class(Moirai2Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_moirai2_panel_predict():
    """Test Moirai2Forecaster fit/predict on panel data without exogenous."""

    data = _make_hierarchical(
        (3, 1), n_columns=1, max_timepoints=20, min_timepoints=20
    )
    data = data.droplevel(1)
    y = data["c0"].to_frame()

    forecaster = Moirai2Forecaster(
        checkpoint_path="Salesforce/moirai-2.0-R-small",
        context_length=16,
    )
    fh = [1, 2, 3]
    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict(fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred.index.get_level_values(-1).unique()) == len(fh)


@pytest.mark.skipif(
    not run_test_for_class(Moirai2Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_moirai2_panel_predict_with_X():
    """Test Moirai2Forecaster fit/predict on panel data with exogenous features."""

    data = _make_hierarchical(
        (3, 1), n_columns=2, max_timepoints=20, min_timepoints=20
    )
    data = data.droplevel(1)
    y = data["c0"].to_frame()
    X = data["c1"].to_frame()

    forecaster = Moirai2Forecaster(
        checkpoint_path="Salesforce/moirai-2.0-R-small",
        context_length=16,
    )
    fh = [1, 2, 3]
    n_train = len(y.index.get_level_values(-1).unique()) - len(fh)
    y_train = y.groupby(level=0, group_keys=False).apply(lambda g: g.iloc[:n_train])
    X_train = X.groupby(level=0, group_keys=False).apply(lambda g: g.iloc[:n_train])
    X_test = X.groupby(level=0, group_keys=False).apply(lambda g: g.iloc[n_train:])

    forecaster.fit(y_train, X=X_train, fh=fh)
    y_pred = forecaster.predict(fh, X=X_test)

    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred.index.get_level_values(-1).unique()) == len(fh)
