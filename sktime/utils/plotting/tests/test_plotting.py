#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

from sktime.forecasting.all import load_airline, ForecastingHorizon
from sktime.forecasting.all import ThetaForecaster, temporal_train_test_split
from sktime.utils.plotting import plot_series
import random
import pandas as pd


def test_plot_series_temporal():
    # Test with temporal indices
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=36)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = ThetaForecaster(sp=12)
    forecaster.fit(y_train)
    y_pred, pred_int = forecaster.predict(fh, return_pred_int=True, alpha=0.05)
    fig, ax = plot_series(
        y_train,
        y_test,
        y_pred,
        labels=["y_train", "y_test", "y_pred"],
        pred_int=pred_int,
    )


def test_plot_series_integer():
    # Test numeric indices:
    X = [float(x + random.randint(1, 10)) for x in range(0, 100)]
    X = pd.Series(X)
    X.index = X.index.astype(int)
    X_test = [x for x in range(1, 21)]
    model = ThetaForecaster()
    model.fit(X, fh=20)
    y_pred, pred_int = model.predict(X_test, return_pred_int=True)
    fig, ax = plot_series(X, y_pred, pred_int=pred_int)
