#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for ForecastX compositor."""

__author__ = ["fkiraly", "yarnabrina"]
__all__ = []

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVR

from sktime.datasets import load_longley
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.compose import ForecastX, make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.var import VAR
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class([ForecastX, VAR, SARIMAX]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_forecastx_logic():
    """Test that ForecastX logic is as expected, compared to manual execution."""
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.split import temporal_train_test_split

    # test case: using pipeline execution
    y, X = load_longley()
    y_train, _, X_train, X_test = temporal_train_test_split(y, X, test_size=3)
    fh = ForecastingHorizon([1, 2, 3])
    columns = ["ARMED", "POP"]

    # ForecastX
    pipe = ForecastX(
        forecaster_X=VAR(),
        forecaster_y=SARIMAX(),
        columns=columns,
    )
    pipe = pipe.fit(y_train, X=X_train, fh=fh)
    # dropping ["ARMED", "POP"] = columns where we expect not to have future values
    y_pred = pipe.predict(fh=fh, X=X_test.drop(columns=columns))

    # comparison case: manual execution
    # fit y forecaster
    arima = SARIMAX().fit(y_train, X=X_train)

    # fit and predict X forecaster
    var = VAR()
    var.fit(X_train[columns])
    var_pred = var.predict(fh)

    # predict y forecaster with predictions from VAR
    X_pred = pd.concat([X_test.drop(columns=columns), var_pred], axis=1)
    y_pred_manual = arima.predict(fh=fh, X=X_pred)

    # compare that test and comparison case results are equal
    assert np.allclose(y_pred, y_pred_manual)


@pytest.mark.skipif(
    not run_test_for_class([ForecastX, ARIMA]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_forecastx_fit_behavior():
    from sktime.forecasting.compose import ForecastX
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

    pipe = ForecastX(
        forecaster_X=NaiveForecaster(),
        forecaster_y=ARIMA(),
    )
    pipe = pipe.fit(y_train, X=X_train, fh=y_test.index)
    y_pred_forecast_X_use_gt = pipe.predict(fh=y_test.index)

    naive = NaiveForecaster()
    naive.fit(X_train)
    x_pred_train = naive.predict(fh=X_train.index)
    arima = ARIMA()
    arima.fit(y_train, X_train)

    y_pred = arima.predict(fh=y_test.index, X=naive.predict(fh=y_test.index))

    pd.testing.assert_series_equal(y_pred_forecast_X_use_gt, y_pred)

    pipe = ForecastX(
        forecaster_X=NaiveForecaster(),
        forecaster_y=ARIMA(),
        fit_behaviour="use_forecast",
    )
    pipe = pipe.fit(y_train, X=X_train, fh=y_test.index)
    y_pred_forecast_X_use_forecast = pipe.predict(fh=y_test.index)

    arima = ARIMA()
    arima.fit(y_train, x_pred_train)
    y_pred = arima.predict(fh=y_test.index, X=naive.predict(fh=y_test.index))

    pd.testing.assert_series_equal(y_pred_forecast_X_use_forecast, y_pred)


@pytest.mark.skipif(
    not run_test_for_class(ForecastX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_forecastx_attrib_broadcast():
    """Test ForecastX broadcasting and forecaster attributes."""
    from sktime.forecasting.compose import ForecastX
    from sktime.forecasting.naive import NaiveForecaster

    df = pd.DataFrame(
        {
            "a": ["series_1", "series_1", "series_1"],
            "b": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "c": [1, 2, 3],
            "d": [4, 5, 6],
            "e": [7, 8, 9],
        }
    )
    df = df.set_index(["a", "b"])

    model = ForecastX(NaiveForecaster(), NaiveForecaster())

    model_1 = model.clone()
    model_1.fit(df[["c"]], X=df[["d", "e"]], fh=[1, 2, 3])

    assert not hasattr(model_1, "forecaster_X_")

    assert hasattr(model_1, "forecaster_y_")
    assert isinstance(model_1.forecaster_y_, NaiveForecaster)
    assert model_1.forecaster_y_.is_fitted

    model_2 = model.clone()
    model_2.fit(df[["c", "d"]], X=df[["e"]], fh=[1, 2, 3])

    assert not hasattr(model_2, "forecaster_X_")

    assert hasattr(model_2, "forecaster_y_")
    assert isinstance(model_2.forecaster_y_, NaiveForecaster)
    assert model_2.forecaster_y_.is_fitted


@pytest.mark.skipif(
    not run_test_for_class(ForecastX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_forecastx_skip_forecaster_X_fitting_logic():
    """Test that ForecastX does not fit forecaster_X, if forecaster_y ignores X"""
    from sklearn.linear_model import LinearRegression

    from sktime.forecasting.compose import ForecastX, YfromX

    y, X = load_longley()

    fh = [1, 2, 3]

    model_supporting_exogenous = YfromX(LinearRegression())
    model_ignoring_exogenous = NaiveForecaster()

    model_1 = ForecastX(
        model_supporting_exogenous.clone(), model_supporting_exogenous.clone()
    )
    model_2 = ForecastX(
        model_supporting_exogenous.clone(), model_ignoring_exogenous.clone()
    )
    model_3 = ForecastX(
        model_ignoring_exogenous.clone(), model_supporting_exogenous.clone()
    )
    model_4 = ForecastX(
        model_ignoring_exogenous.clone(), model_ignoring_exogenous.clone()
    )

    assert hasattr(model_1, "forecaster_y")
    assert hasattr(model_2, "forecaster_y")
    assert hasattr(model_3, "forecaster_y")
    assert hasattr(model_4, "forecaster_y")

    assert hasattr(model_1, "forecaster_X")
    assert hasattr(model_2, "forecaster_X")
    assert hasattr(model_3, "forecaster_X")
    assert hasattr(model_4, "forecaster_X")

    assert not hasattr(model_1, "forecaster_y_")
    assert not hasattr(model_2, "forecaster_y_")
    assert not hasattr(model_3, "forecaster_y_")
    assert not hasattr(model_4, "forecaster_y_")

    assert not hasattr(model_1, "forecaster_X_")
    assert not hasattr(model_2, "forecaster_X_")
    assert not hasattr(model_3, "forecaster_X_")
    assert not hasattr(model_4, "forecaster_X_")

    model_1.fit(y, X=X, fh=fh)
    model_2.fit(y, X=X, fh=fh)
    model_3.fit(y, X=X, fh=fh)
    model_4.fit(y, X=X, fh=fh)

    assert hasattr(model_1, "forecaster_y_")
    assert hasattr(model_2, "forecaster_y_")
    assert hasattr(model_3, "forecaster_y_")
    assert hasattr(model_4, "forecaster_y_")

    assert model_1.forecaster_y_.is_fitted
    assert model_2.forecaster_y_.is_fitted
    assert model_3.forecaster_y_.is_fitted
    assert model_4.forecaster_y_.is_fitted

    assert hasattr(model_1, "forecaster_X_")
    assert hasattr(model_2, "forecaster_X_")
    assert not hasattr(model_3, "forecaster_X_")
    assert not hasattr(model_4, "forecaster_X_")

    assert model_1.forecaster_X_.is_fitted
    assert model_2.forecaster_X_.is_fitted


@pytest.mark.skipif(
    not run_test_for_class(ForecastX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasting_algorithm", [make_reduction(SVR(), window_length=2), NaiveForecaster()]
)
@pytest.mark.parametrize(
    "future_unknown_columns",
    [["GNPDEFL", "GNP"], ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP"], None],
)
def test_forecastx_flow_known_unknown_columns(
    forecasting_algorithm, future_unknown_columns
):
    """Test that ForecastX does not fit forecaster_X, if forecaster_y ignores X"""
    from sktime.forecasting.compose import ForecastX

    y, X = load_longley()

    fh = [1, 2]

    y_train_val, y_test, X_train_val, X_test = temporal_train_test_split(
        y, X, test_size=max(fh)
    )
    y_train, y_val, X_train, X_val = temporal_train_test_split(
        y_train_val, X_train_val, test_size=max(fh)
    )

    model = ForecastX(
        forecasting_algorithm.clone(),
        forecasting_algorithm.clone(),
        columns=future_unknown_columns,
    )

    assert hasattr(model, "forecaster_y")
    assert hasattr(model, "forecaster_X")

    assert not hasattr(model, "forecaster_y_")
    assert not hasattr(model, "forecaster_X_")

    model.fit(y_train, X=X_train, fh=fh)

    assert hasattr(model, "forecaster_y_")
    assert model.forecaster_y_.is_fitted

    if model.get_tag("ignores-exogeneous-X"):
        assert not hasattr(model, "forecaster_X_")
    else:
        assert hasattr(model, "forecaster_X_")
        assert model.forecaster_X_.is_fitted

    y_val_pred = model.predict(X=X_test)
    np.testing.assert_array_equal(y_val.index, y_val_pred.index)

    model.update(y_val, X=X_val)

    y_test_pred = model.predict(X=X_test)
    np.testing.assert_array_equal(y_test.index, y_test_pred.index)


@pytest.mark.skipif(
    not run_test_for_class(ForecastX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_forecastx_exog_for_forecaster_x():
    """Test that ForecastX forecaster_X uses exogenous data as told by parameter."""
    from sklearn.linear_model import LinearRegression

    from sktime.forecasting.compose import ForecastX, YfromX

    y, X = load_longley()

    fh = [1, 2, 3]

    model_supporting_exogenous = YfromX(LinearRegression())

    cols_to_forecast = ["GNPDEFL", "GNP"]

    model_1 = ForecastX(
        model_supporting_exogenous.clone(),
        model_supporting_exogenous.clone(),
        columns=cols_to_forecast,
        forecaster_X_exogeneous="None",
    )

    model_1.fit(y, X=X, fh=fh)
    assert model_1.forecaster_X_._X is None

    model_2 = ForecastX(
        model_supporting_exogenous.clone(),
        model_supporting_exogenous.clone(),
        columns=cols_to_forecast,
        forecaster_X_exogeneous="complement",
    )

    model_2.fit(y, X=X, fh=fh)
    assert model_2.forecaster_X_._X.columns.tolist() == ["UNEMP", "ARMED", "POP"]

    model_3 = ForecastX(
        model_supporting_exogenous.clone(),
        model_supporting_exogenous.clone(),
        columns=cols_to_forecast,
        forecaster_X_exogeneous=["UNEMP", "ARMED"],
    )

    model_3.fit(y, X=X, fh=fh)
    assert model_3.forecaster_X_._X.columns.tolist() == ["UNEMP", "ARMED"]


@pytest.mark.skipif(
    not run_test_for_class([ForecastX, ARIMA]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("predict_behaviour_option", ["use_forecasts", "use_actuals"])
def test_use_of_passed_unknown_X(predict_behaviour_option: str) -> None:
    from sktime.forecasting.compose import ForecastX

    y, X = load_longley()
    fh = [1, 2, 3, 4]
    cols_to_forecast = ["GNPDEFL", "GNP"]

    y_train, _, X_train, X_test = temporal_train_test_split(y, X, test_size=4)

    model_with_explicit_columns = ForecastX(
        ARIMA(),
        forecaster_X=NaiveForecaster(),
        columns=cols_to_forecast,
        predict_behaviour=predict_behaviour_option,
    )

    model_with_explicit_columns.fit(y_train, X=X_train, fh=fh)

    with mock.patch.object(
        model_with_explicit_columns.forecaster_X_, "predict"
    ) as mock_predict:
        mock_predict.return_value = X_test

        _ = model_with_explicit_columns.predict(X=X_test.drop(columns=cols_to_forecast))

        mock_predict.assert_called_once()

    with mock.patch.object(
        model_with_explicit_columns.forecaster_X_, "predict"
    ) as mock_predict:
        mock_predict.return_value = X_test

        _ = model_with_explicit_columns.predict(X=X_test)

        if predict_behaviour_option == "use_forecasts":
            mock_predict.assert_called_once()
        elif predict_behaviour_option == "use_actuals":
            mock_predict.assert_not_called()

    model_with_implicit_columns = ForecastX(
        ARIMA(),
        forecaster_X=NaiveForecaster(),
        predict_behaviour=predict_behaviour_option,
    )

    model_with_implicit_columns.fit(y_train, X=X_train, fh=fh)

    with mock.patch.object(
        model_with_implicit_columns.forecaster_X_, "predict"
    ) as mock_predict:
        mock_predict.return_value = X_test

        _ = model_with_implicit_columns.predict()

        mock_predict.assert_called_once()

    with mock.patch.object(
        model_with_implicit_columns.forecaster_X_, "predict"
    ) as mock_predict:
        mock_predict.return_value = X_test

        _ = model_with_implicit_columns.predict(X=X_test)

        if predict_behaviour_option == "use_forecasts":
            mock_predict.assert_called_once()
        elif predict_behaviour_option == "use_actuals":
            mock_predict.assert_not_called()


@pytest.mark.skipif(
    not run_test_for_class([ForecastX, ARIMA]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("cols_to_forecast", [["GNPDEFL", "GNP"], ["ARMED", "POP"]])
def test_forecaster_X_exogeneous(cols_to_forecast):
    """Test that ForecastX forecaster_X uses exogenous data as told by parameter."""
    from sktime.forecasting.compose import ForecastX
    from sktime.split import temporal_train_test_split

    y, X = load_longley()

    fh = [1, 2, 3, 4]
    y_train, _, X_train, X_test = temporal_train_test_split(y, X, test_size=max(fh))

    forecaster = ARIMA()
    pipeline1 = ForecastX(
        forecaster.clone(),
        forecaster_X=forecaster.clone(),
        columns=cols_to_forecast,
        forecaster_X_exogeneous="complement",
    )

    pipeline1.fit(y_train, X=X_train, fh=fh)
    y_pred1 = pipeline1.predict(X=X_test.drop(columns=cols_to_forecast))

    pipeline2 = ForecastX(
        forecaster.clone(),
        forecaster_X=forecaster.clone(),
        columns=cols_to_forecast,
        forecaster_X_exogeneous="None",
    )

    pipeline2.fit(y_train, X=X_train, fh=fh)
    y_pred2 = pipeline2.predict(X=X_test.drop(columns=cols_to_forecast))
    np.testing.assert_array_equal(y_pred1.index, y_pred2.index)
