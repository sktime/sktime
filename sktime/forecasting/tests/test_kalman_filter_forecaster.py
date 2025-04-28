# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Kalman Filter forecaster unit tests."""

__author__ = ["NoaBenAmi"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.kalman_filter_forecaster import (
    KalmanFilterForecaster,
    _forecast_matrices,
    _split,
)
from sktime.transformations.series.kalman_filter import KalmanFilterTransformerFP
from sktime.transformations.series.tests.test_kalman_filter import create_data

# ts stands for time steps
ts = 10


def _gen_params(max_horizon, dim, params=None):
    p = {
        "state_transition": create_data((max_horizon, dim, dim)),
        "control_transition": None,
        "process_noise": create_data((dim, dim)),
        "measurement_noise": create_data((dim, dim)),
        "initial_state": create_data(dim),
        "initial_state_covariance": create_data((dim, dim)),
        "denoising": False,
    }
    if params is not None:
        for key in params.keys():
            p[key] = params[key]
    return p


def _in_sample_and_future_pred_indices(fh, cutoff):
    fh_relative = fh.to_relative(cutoff)
    return [i - 1 for i in fh_relative[fh_relative < 1]], [
        i - 1 for i in fh_relative[fh_relative > 0]
    ]


indices = pd.PeriodIndex(pd.date_range("2000-01", periods=ts, freq="M"))
# Forecast Horizons
fh1 = ForecastingHorizon(
    pd.PeriodIndex(pd.date_range(indices[-3].to_timestamp(), periods=ts, freq="M")),
    is_relative=False,
)
fh2 = ForecastingHorizon([-9, -3, 0, 2, 3])
fh3 = ForecastingHorizon([1, 5, 8])

# Data and Params
# Data1
data1 = pd.DataFrame(
    data=create_data(shape=(ts, 3), missing_values=True), index=indices
)
params_1_1 = _gen_params(ts + 8, 3)
params_1_2 = _gen_params(ts + 3, 3, {"control_transition": create_data((ts + 3, 3, 2))})
# Data2
data2 = pd.DataFrame(
    data=create_data(shape=(ts, 1), missing_values=True), index=indices
)
# Data3
data3 = pd.DataFrame(data=create_data(shape=(ts, 2)), index=indices)


def _transformer_predictions(
    measurements,
    indices,
    us,
    state_transition=None,
    control_transition=None,
    process_noise=None,
    measurement_noise=None,
    measurement_function=None,
    initial_state=None,
    initial_state_covariance=None,
    estimate_matrices=None,
    denoising=False,
):
    transformer = KalmanFilterTransformerFP(
        state_dim=measurements.shape[-1],
        state_transition=state_transition,
        control_transition=control_transition,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        measurement_function=measurement_function,
        initial_state=initial_state,
        initial_state_covariance=initial_state_covariance,
        estimate_matrices=estimate_matrices,
        denoising=denoising,
    )

    x_transform = transformer.fit_transform(X=measurements, y=us)

    predictions = [p for p in x_transform[indices]]
    return predictions, x_transform[-1], transformer


def _generate_transformer_predictions(
    params, measurements, in_samples_indices, future_preds_indices, us
):
    time_steps, state_dim = measurements.shape
    us_fit, us_predict = _split(us, time_steps)
    params["state_transition"], F_forecast = _split(
        params["state_transition"], time_steps
    )
    params["control_transition"], B_forecast = _split(
        params["control_transition"], time_steps
    )

    in_samples_predictions, x_init, in_samples_transformer = _transformer_predictions(
        measurements.to_numpy(), in_samples_indices, us_fit, **params
    )

    if len(future_preds_indices) == 0:
        return np.array(in_samples_predictions)

    Fs = _forecast_matrices(F_forecast, in_samples_transformer.F_, time_steps - 1)
    Bs = _forecast_matrices(B_forecast, params["control_transition"], time_steps - 1)

    future_predictions, _, _ = _transformer_predictions(
        np.full((future_preds_indices[-1] + 1, state_dim), np.nan),
        future_preds_indices,
        us_predict,
        state_transition=Fs,
        control_transition=Bs,
        initial_state=x_init,
    )

    return np.array(in_samples_predictions + future_predictions)


@pytest.mark.parametrize(
    "params, fh, measurements, us",
    [(params_1_1, fh3, data1, None), (params_1_2, fh2, data1, None)],
)
def test_forecaster(params, fh, measurements, us):
    """Test KalmanFilterForecaster `fit` and `predict`.

    Comparing results of forecaster against two instances of KalmanFilterTransformerFP.
    One instance for in-samples predictions and one for future predictions.
    """
    params = dict(params)
    in_samples_indices, future_preds_indices = _in_sample_and_future_pred_indices(
        fh=fh, cutoff=measurements.index[-1]
    )

    forecaster = KalmanFilterForecaster(**params)
    y_forecast = forecaster.fit_predict(y=measurements, X=us, fh=fh)
    expected = _generate_transformer_predictions(
        params, measurements, in_samples_indices, future_preds_indices, us
    )

    assert np.array_equal(y_forecast.to_numpy(), expected)
