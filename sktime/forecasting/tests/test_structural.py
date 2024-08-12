"""UnobservedComponents Tests."""

__author__ = ["juanitorduz"]

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.structural import UnobservedComponents
from sktime.split import SlidingWindowSplitter
from sktime.tests.test_switch import run_test_for_class


class ModelSpec:
    """Model specification."""

    def __init__(self, level, params):
        self.level = level
        self.params = params

    def generate_parametrized_sample_data(
        self, n, sigma_zeta=0.1, sigma_eta=0.1, sigma_epsilon=0.1
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
            zeta[t] = self.params["zeta"] * np.random.normal(loc=0.0, scale=sigma_zeta)
            beta[t] = self.params["beta_1"] * beta[t - 1] + zeta[t]

            eta[t] = self.params["eta"] * np.random.normal(loc=0.0, scale=sigma_eta)
            mu[t] = (
                self.params["mu_1"] * mu[t - 1]
                + self.params["beta_1"] * beta[t - 1]
                + eta[t]
            )

            epsilon[t] = self.params["epsilon"] * np.random.normal(
                loc=0.0, scale=sigma_epsilon
            )
            y[t] = self.params["mu"] * mu[t] + epsilon[t]

        return y, mu, beta


# Parameters to generate sample data depending of the model type.
MODELS = [
    ModelSpec(
        level="irregular",
        params={
            "zeta": 0,
            "beta": 0,
            "beta_1": 0,
            "eta": 0,
            "epsilon": 1,
            "mu_1": 0,
            "mu": 0,
        },
    ),
    ModelSpec(
        level="fixed intercept",
        params={
            "zeta": 0,
            "beta": 0,
            "beta_1": 0,
            "eta": 0,
            "epsilon": 0,
            "mu_1": 0,
            "mu": 1,
        },
    ),
    ModelSpec(
        level="random walk",
        params={
            "zeta": 0,
            "beta": 0,
            "beta_1": 0,
            "eta": 1,
            "epsilon": 0,
            "mu_1": 1,
            "mu": 1,
        },
    ),
    ModelSpec(
        level="local level",
        params={
            "zeta": 0,
            "beta": 0,
            "beta_1": 0,
            "eta": 1,
            "epsilon": 1,
            "mu_1": 1,
            "mu": 1,
        },
    ),
    ModelSpec(
        level="smooth trend",
        params={
            "zeta": 1,
            "beta": 0,
            "beta_1": 1,
            "eta": 0,
            "epsilon": 1,
            "mu_1": 1,
            "mu": 1,
        },
    ),
    ModelSpec(
        level="local linear trend",
        params={
            "zeta": 1,
            "beta": 0,
            "beta_1": 1,
            "eta": 1,
            "epsilon": 1,
            "mu_1": 1,
            "mu": 1,
        },
    ),
]


@pytest.fixture(params=MODELS, ids=[m.level for m in MODELS])
def level_sample_data(request):
    """Generate sample data for a given model specification.

    We add external regressors and a seasonality component.
    """
    min_date = pd.to_datetime("2017-01-01")
    max_date = pd.to_datetime("2022-01-01")

    data_df = pd.DataFrame(
        data={"date": pd.date_range(start=min_date, end=max_date, freq="M")}
    )

    n = data_df.shape[0]
    model_spec = request.param
    y, _, _ = model_spec.generate_parametrized_sample_data(n=n)

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
    return model_spec.level, data_df


@pytest.fixture
def level_sample_data_split(level_sample_data):
    """Split sample data into train and test sets."""
    train_test_ratio = 0.80
    level, data_df = level_sample_data
    n = data_df.shape[0]
    n_train = int(n * train_test_ratio)
    n_test = n - n_train

    data_train_df = data_df[:n_train]
    data_test_df = data_df[-n_test:]

    y_train = data_train_df["z"]
    X_train = data_train_df[["x"]]
    X_test = data_test_df[["x"]]
    fh = np.arange(X_test.shape[0]) + 1
    return level, y_train, X_train, X_test, fh


@pytest.fixture
def y_airlines():
    """Load sample data from the airlines dataset."""
    return load_airline()


@pytest.mark.skipif(
    not run_test_for_class(UnobservedComponents),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("level", [m.level for m in MODELS])
@pytest.mark.parametrize("fh_length", [1, 3, 5, 10, 20])
def test_results_consistency(level, fh_length, y_airlines):
    """Check consistency between wrapper and statsmodels original implementation."""
    from statsmodels.tsa.statespace.structural import (
        UnobservedComponents as _UnobservedComponents,
    )

    fh = np.arange(fh_length) + 1
    # Fit and predict with forecaster.
    forecaster = UnobservedComponents(level=level)
    forecaster.fit(y_airlines)
    y_pred_forecaster = forecaster.predict(fh=fh)
    # Fit train statsmodels original model.
    model = _UnobservedComponents(level=level, endog=y_airlines)
    result = model.fit(disp=0)
    y_pred_base = result.forecast(steps=fh_length)
    y_pred_base.name = y_airlines.name
    assert_series_equal(left=y_pred_forecaster, right=y_pred_base)
    assert len(fh) == y_pred_forecaster.shape[0]


@pytest.mark.skipif(
    not run_test_for_class(UnobservedComponents),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_result_consistency_exog(level_sample_data_split):
    """Check consistency between wrapper and statsmodels original implementation.

    We add external regressors and a seasonality component.
    """
    from statsmodels.tsa.statespace.structural import (
        UnobservedComponents as _UnobservedComponents,
    )

    level, y_train, X_train, X_test, fh = level_sample_data_split

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
    n_test = X_test.shape[0]
    y_pred_base = result.forecast(steps=n_test, exog=X_test)
    y_pred_base.name = y_train.name
    assert_series_equal(left=y_pred_forecaster, right=y_pred_base)
    assert len(fh) == y_pred_forecaster.shape[0]


@pytest.mark.skipif(
    not run_test_for_class(UnobservedComponents),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("alpha", [0.01, 0.05, [0.01, 0.05]])
@pytest.mark.parametrize("coverage", [0.6, 0.99, [0.9, 0.95]])
@pytest.mark.parametrize("fh_length", [1, 3, 5, 10, 20])
def test_prediction_intervals_no_exog(alpha, coverage, fh_length, y_airlines):
    """Test prediction intervals when no exogenous regressors are present."""
    fh = np.arange(fh_length) + 1
    forecaster = UnobservedComponents()
    forecaster.fit(y_airlines)
    quantiles_df = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    intervals_df = forecaster.predict_interval(fh=fh, coverage=coverage)
    if isinstance(alpha, list):
        assert quantiles_df.shape == (fh_length, len(alpha))
    else:
        assert quantiles_df.shape == (fh_length, 1)

    if isinstance(coverage, list):
        assert intervals_df.shape == (fh_length, 2 * len(coverage))
    else:
        intervals_np = intervals_df.to_numpy().flatten()
        assert intervals_df.shape == (fh_length, 2)
        assert intervals_np[0] < intervals_np[1]


@pytest.mark.skipif(
    not run_test_for_class(UnobservedComponents),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("alpha", [0.01, 0.05, [0.01, 0.05]])
@pytest.mark.parametrize("coverage", [0.6, 0.99, [0.9, 0.95]])
def test_prediction_intervals_exog(alpha, coverage, level_sample_data_split):
    """Test prediction intervals when exogenous regressors are present."""
    level, y_train, X_train, X_test, fh = level_sample_data_split
    fh_length = X_test.shape[0]
    model_spec = {
        "level": level,
        "freq_seasonal": [{"period": 12, "harmonics": 4}],
        "autoregressive": 1,
        "mle_regression": False,
    }
    forecaster = UnobservedComponents(**model_spec)
    forecaster.fit(y=y_train, X=X_train)
    quantiles_df = forecaster.predict_quantiles(fh=fh, X=X_test, alpha=alpha)
    intervals_df = forecaster.predict_interval(fh=fh, X=X_test, coverage=coverage)

    if isinstance(alpha, list):
        assert quantiles_df.shape == (fh_length, len(alpha))
    else:
        assert quantiles_df.shape == (fh_length, 1)

    if isinstance(coverage, list):
        assert intervals_df.shape == (fh_length, 2 * len(coverage))
    else:
        intervals_np = intervals_df.to_numpy().flatten()
        assert intervals_df.shape == (fh_length, 2)
        assert intervals_np[0] < intervals_np[1]


@pytest.mark.skipif(
    not run_test_for_class(UnobservedComponents),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_evaluate_exog():
    """Test evaluate works when exogenous regressors are present."""
    y, X = load_longley()
    forecaster = UnobservedComponents(level="local linear trend")
    cv = SlidingWindowSplitter(fh=[1, 2, 3], window_length=4, step_length=1)
    results = evaluate(
        forecaster=forecaster,
        y=y,
        X=X,
        cv=cv,
        strategy="refit",
        return_data=True,
        error_score="raise",
    )
    assert results.shape == (10, 8)
