# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Forecaster interface to the TabICL tabular foundation model.

Uses a sliding window reduction to convert time series forecasting into
a tabular regression problem, solved by TabICLRegressor.

References
----------
.. [1] Qu, J., Holzmüller, D., Varoquaux, G., Le Morvan, M. (2025).
   TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.
   ICML 2025. https://arxiv.org/abs/2502.05564
.. [2] Qu, J., Holzmüller, D., Varoquaux, G., Le Morvan, M. (2026).
   TabICLv2: A better, faster, scalable, and open tabular foundation model.
   https://arxiv.org/abs/2602.11139
"""

__author__ = ["Anamika457"]
__all__ = ["TabICLForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class TabICLForecaster(BaseForecaster):
    """Time series forecaster using the TabICL tabular foundation model.

    Wraps ``TabICLRegressor`` from the ``tabicl`` package using a sliding
    window reduction. Each window of ``window_length`` past values becomes
    a row of tabular features, with the next value as the target.

    At predict time, forecasts are made recursively: each predicted value
    is appended to the window and used as input for the next step.

    This mirrors the relationship between TabPFN and TabPFN-TS in sktime.
    TabICLv2 outperforms TabPFN-2.5 on TabArena benchmarks at comparable
    or faster runtimes, so we can expect similar gains in the time series
    setting.

    Parameters
    ----------
    window_length : int, default=10
        Number of past time steps used as lag features. Larger values give
        more context but increase runtime, since TabICL scales as O(n^2)
        in the number of training rows.
    n_estimators : int, default=8
        Number of ensemble members in TabICL. Higher values improve accuracy
        at the cost of runtime. Passed directly to ``TabICLRegressor``.
    use_kv_cache : bool, default=True
        If True, enables KV-caching during fit. TabICL's fit is essentially
        free - the transformer forward pass happens in predict. Without
        caching, every recursive predict step re-encodes the full training
        context. Enabling this trades memory for speed.
    device : str or None, default=None
        Inference device. None auto-selects CUDA if available, else CPU.
    checkpoint_version : str, default="tabicl-regressor-v2-20260212.ckpt"
        TabICL checkpoint to load. See tabicl docs for available versions.
    random_state : int, default=42
        Random seed for reproducibility, passed to ``TabICLRegressor``.
    verbose : bool, default=False
        If True, TabICL prints progress during inference.

    Examples
    --------
    >>> from sktime.forecasting.tabicl import TabICLForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TabICLForecaster(window_length=12, n_estimators=4)
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    TabICLForecaster(...)
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    """

    _tags = {
        "authors": "Anamika457",
        "maintainers": "Anamika457",
        "python_dependencies": "tabicl>=2.0.0",
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "capability:insample": False,
        "capability:random_state": True,
    }

    def __init__(
        self,
        window_length=10,
        n_estimators=8,
        use_kv_cache=True,
        device=None,
        checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
        random_state=42,
        verbose=False,
    ):
        self.window_length = window_length
        self.n_estimators = n_estimators
        self.use_kv_cache = use_kv_cache
        self.device = device
        self.checkpoint_version = checkpoint_version
        self.random_state = random_state
        self.verbose = verbose

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster.

        Parameters
        ----------
        y : pd.Series
            Time series to fit on.
        X : pd.DataFrame, ignored
            Not used, present for API compatibility.
        fh : ForecastingHorizon or None
            Can be passed here or later in predict.

        Returns
        -------
        self
        """
        from tabicl import TabICLRegressor

        wl = self.window_length
        values = y.values

        if len(values) <= wl:
            raise ValueError(
                f"Training series length ({len(values)}) must be greater than "
                f"window_length ({wl}). Reduce window_length or provide more data."
            )

        X_tab, y_tab = _make_tabular(values, wl)
        X_tab = _impute_array(X_tab)

        self.regressor_ = TabICLRegressor(
            n_estimators=self.n_estimators,
            device=self.device,
            checkpoint_version=self.checkpoint_version,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.regressor_.fit(X_tab, y_tab, kv_cache=self.use_kv_cache)
        self._last_window = values[-wl:].copy()

        return self

    def _predict(self, fh, X=None):
        """Make forecasts for fh steps ahead.

        Parameters
        ----------
        fh : ForecastingHorizon
        X : pd.DataFrame, ignored

        Returns
        -------
        y_pred : pd.Series
        """
        fh_rel = fh.to_relative(self.cutoff)
        max_step = int(max(fh_rel))

        window = self._last_window.copy()
        predictions = {}

        for step in range(1, max_step + 1):
            x_row = window[-self.window_length :].reshape(1, -1)
            y_hat = self.regressor_.predict(_impute_array(x_row))[0]

            if step in fh_rel:
                predictions[step] = y_hat

            window = np.append(window, y_hat)

        fh_abs = fh.to_absolute(self.cutoff)
        return pd.Series(
            [predictions[s] for s in fh_rel],
            index=fh_abs.to_pandas(),
            name=self._y.name,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default "default"

        Returns
        -------
        params : list of dict
        """
        params1 = {
            "window_length": 3,
            "n_estimators": 1,
            "use_kv_cache": False,
        }
        params2 = {
            "window_length": 5,
            "n_estimators": 2,
            "use_kv_cache": True,
        }
        return [params1, params2]


def _make_tabular(values, window_length):
    """Slide a window over values to produce tabular (X, y) pairs.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
    window_length : int

    Returns
    -------
    X : np.ndarray, shape (n - window_length, window_length)
    y : np.ndarray, shape (n - window_length,)
    """
    n = len(values)
    X = np.empty((n - window_length, window_length), dtype=np.float64)
    y = np.empty(n - window_length, dtype=np.float64)
    for i in range(n - window_length):
        X[i] = values[i : i + window_length]
        y[i] = values[i + window_length]
    return X, y


def _impute_array(X):
    """Column-mean impute a 2-D numpy array in place of NaNs.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    X : np.ndarray, NaNs replaced with column means (0 if column is all NaN)
    """
    if not np.isnan(X).any():
        return X
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X
