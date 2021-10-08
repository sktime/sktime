# -*- coding: utf-8 -*-
"""UnobservedComponents Tests."""
__author__ = ["juanitorduz"]

import numpy as np
import pandas as pd
from sktime.forecasting.structural import UnobservedComponents, _UnobservedComponents
from sktime.datasets import load_airline
from pandas.testing import assert_series_equal


# Parameters to generate sample data depending of the model type.
structure_params = {
    "irregular": {
        "zeta": 0,
        "beta": 0,
        "beta_1": 0,
        "eta": 0,
        "epsilon": 1,
        "mu_1": 0,
        "mu": 0,
    },
    "fixed intercept": {
        "zeta": 0,
        "beta": 0,
        "beta_1": 0,
        "eta": 0,
        "epsilon": 0,
        "mu_1": 0,
        "mu": 1,
    },
    "random walk": {
        "zeta": 0,
        "beta": 0,
        "beta_1": 0,
        "eta": 1,
        "epsilon": 0,
        "mu_1": 1,
        "mu": 1,
    },
    "local level": {
        "zeta": 0,
        "beta": 0,
        "beta_1": 0,
        "eta": 1,
        "epsilon": 1,
        "mu_1": 1,
        "mu": 1,
    },
    "smooth trend": {
        "zeta": 1,
        "beta": 0,
        "beta_1": 1,
        "eta": 0,
        "epsilon": 1,
        "mu_1": 1,
        "mu": 1,
    },
    "local linear trend": {
        "zeta": 1,
        "beta": 0,
        "beta_1": 1,
        "eta": 1,
        "epsilon": 1,
        "mu_1": 1,
        "mu": 1,
    },
}

levels = structure_params.keys()


def generate_parametrized_sample_data(
    n, params, sigma_zeta=0.1, sigma_eta=0.1, sigma_epsilon=0.1
):
    """Generate sample data given the state model specification."""
    # Initialize variables
    y = np.zeros(n)
    mu = np.zeros(n)
    beta = np.zeros(n)
    epsilon = np.zeros(n)
    eta = np.zeros(n)
    zeta = np.zeros(n)

    # Sample from model parameters.
    for t in range(1, n):

        zeta[t] = params["zeta"] * np.random.normal(loc=0.0, scale=sigma_zeta)
        beta[t] = params["beta_1"] * beta[t - 1] + zeta[t]

        eta[t] = params["eta"] * np.random.normal(loc=0.0, scale=sigma_eta)
        mu[t] = params["mu_1"] * mu[t - 1] + params["beta_1"] * beta[t - 1] + eta[t]

        epsilon[t] = params["epsilon"] * np.random.normal(loc=0.0, scale=sigma_epsilon)
        y[t] = params["mu"] * mu[t] + epsilon[t]

    return y, mu, beta


def generate_sample_data(params):
    """Generate sample data for a given model specification.

    We add external regressors and a seasonality component.
    """
    min_date = pd.to_datetime("2017-01-01")
    max_date = pd.to_datetime("2022-01-01")

    data_df = pd.DataFrame(
        data={"date": pd.date_range(start=min_date, end=max_date, freq="M")}
    )

    n = data_df.shape[0]
    y, _, _ = generate_parametrized_sample_data(n=n, params=params)

    data_df["y"] = y
    # Add external regressor.
    x = np.random.uniform(low=0.0, high=1.0, size=n)
    data_df["x"] = np.where(x > 0.70, x, 0)
    # Add seasonal component.
    data_df["cs"] = np.sin(2 * np.pi * data_df["date"].dt.dayofyear / 356.5)
    data_df["cc"] = np.cos(3 * np.pi * data_df["date"].dt.dayofyear / 356.5)
    data_df["s"] = data_df["cs"] + data_df["cc"]
    # Construct target variable.
    data_df["z"] = data_df["y"] + data_df["x"] + data_df["s"]
    return data_df


def test_results_consistency(levels=levels):
    """Check consistency between wrapper and statsmodels original implementation."""
    y = load_airline()
    fh_length = [3, 5, 10]
    for n in fh_length:
        fh = np.arange(n) + 1
        for level in levels:
            # Fit and predict with forecaster.
            forecaster = UnobservedComponents(level=level)
            forecaster.fit(y)
            y_pred_forecaster = forecaster.predict(fh=fh)
            # Fit train statsmodels original model.
            model = _UnobservedComponents(level=level, endog=y)
            result = model.fit(disp=0)
            y_pred_base = result.forecast(steps=n)
            assert_series_equal(left=y_pred_forecaster, right=y_pred_base)
            assert len(fh) == y_pred_forecaster.shape[0]


def test_result_consistency_exog(structure_params=structure_params, levels=levels):
    """Check consistency between wrapper and statsmodels original implementation.

    We add external regressors and a seasonality component.
    """
    train_test_ratio = 0.80

    for level in levels:
        # Prepare data.
        data_df = generate_sample_data(structure_params[level])
        n = data_df.shape[0]

        n_train = int(n * train_test_ratio)
        n_test = n - n_train

        data_train_df = data_df[:n_train]
        data_test_df = data_df[-n_test:]

        y_train = data_train_df["z"]
        X_train = data_train_df[["x"]]
        X_test = data_test_df[["x"]]
        fh = np.arange(X_test.shape[0]) + 1

        model_spec = {
            "level": level,
            "freq_seasonal": [{"period": 12, "harmonics": 4}],
            "autoregressive": 1,
            "mle_regression": False,
        }
        # Fit and predict with forecaster.
        forecaster = UnobservedComponents(**model_spec)
        forecaster.fit(y=y_train, X=X_train)
        y_pred_forecaster = forecaster.predict(fh=fh, X=X_test)
        # Fit train statsmodels original model.
        model = _UnobservedComponents(endog=y_train, exog=X_train, **model_spec)
        result = model.fit(disp=0)
        y_pred_base = result.forecast(steps=n_test, exog=X_test)
        assert_series_equal(left=y_pred_forecaster, right=y_pred_base)
        assert len(fh) == y_pred_forecaster.shape[0]
