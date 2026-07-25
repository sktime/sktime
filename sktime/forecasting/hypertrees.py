# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hyper-Trees forecasters."""

__author__ = ["aminehd"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class HyperTreeSTLForecaster(BaseForecaster):
    """Hyper-Tree-STL forecaster, from the ``hypertrees-forecasting`` package.

    Direct interface to ``hypertrees.models.HyperTreeSTL`` [1]_.

    Hyper-Trees use a gradient boosted tree (LightGBM) to learn the parameters
    of a classical time series model as functions of features, rather than
    forecasting the series directly. ``HyperTreeSTL`` uses an STL-style
    decomposition (trend plus Fourier seasonal terms) as the target model, so
    the tree predicts the decomposition parameters and the STL model produces
    the forecast.

    The interfaced estimator is univariate and models a single series.

    Parameters
    ----------
    period : int, optional (default=12)
        Seasonal period of the series, e.g. 12 for monthly, 4 for quarterly.
    num_seasonal_components : int, optional (default=1)
        Number of Fourier seasonal harmonics in the decomposition.
    variant : str, optional (default="paper")
        Model variant, passed as ``type`` to the interfaced estimator.
        ``"paper"`` uses the method from the paper, ``"default"`` uses updated
        trend smoothing but requires a forecasting horizon of at least 8 steps.
    freq : str, optional (default=None)
        Pandas frequency string for the internal ``date`` axis. If None, a
        monthly (``"MS"``) axis is used; forecasts are returned on the index
        implied by ``fh``, so this only affects the interfaced Fourier basis.
    lgb_params : dict, optional (default=None)
        LightGBM parameters. If None, ``{"learning_rate": 0.1}`` is used.
    num_iterations : int, optional (default=100)
        Number of boosting rounds.
    seed : int, optional (default=123)
        Random seed for the interfaced estimator.

    References
    ----------
    .. [1] Maerz, Alexander, and Kashif Rasul. "Forecasting with Hyper-Trees."
      arXiv preprint arXiv:2405.07836 (2024).

    Examples
    --------
    >>> from sktime.forecasting.hypertrees import HyperTreeSTLForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = HyperTreeSTLForecaster(period=12)  # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    HyperTreeSTLForecaster(...)
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    """

    _tags = {
        "authors": ["StatMixedML", "kashif", "aminehd"],
        "maintainers": ["aminehd"],
        "python_dependencies": ["hypertrees-forecasting"],
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:exogenous": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "capability:insample": False,
    }

    def __init__(
        self,
        period=12,
        num_seasonal_components=1,
        variant="paper",
        freq=None,
        lgb_params=None,
        num_iterations=100,
        seed=123,
    ):
        self.period = period
        self.num_seasonal_components = num_seasonal_components
        self.variant = variant
        self.freq = freq
        self._freq = freq if freq is not None else "MS"
        self.lgb_params = lgb_params
        self._lgb_params = (
            lgb_params if lgb_params is not None else {"learning_rate": 0.1}
        )
        self.num_iterations = num_iterations
        self.seed = seed
        super().__init__()

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        y : pd.Series
            guaranteed to be univariate.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to.
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.

        Returns
        -------
        self : reference to self
        """
        from hypertrees.models.HyperTreeSTL import HyperTreeSTL

        self._series_id = 0
        self._train_len = len(y)
        fcst_h = int(np.max(fh.to_relative(self.cutoff)._values))
        self._dates = pd.date_range(
            "2000-01-01", periods=self._train_len + fcst_h, freq=self._freq
        )

        times = np.arange(1, self._train_len + 1)
        train_data = pd.DataFrame(
            {
                "series_id": self._series_id,
                "date": self._dates[: self._train_len],
                "value": np.asarray(y.to_numpy(), dtype=float),
                "time": times,
            }
        )
        self._x_cols = []
        if X is not None:
            self._x_cols = list(X.columns)
            for col in self._x_cols:
                train_data[col] = np.asarray(X[col].to_numpy())

        forecaster = HyperTreeSTL(
            period=self.period,
            num_seasonal_components=self.num_seasonal_components,
            freq=self._freq,
            fcst_h=fcst_h,
            type=self.variant,
        )
        forecaster.train(
            lgb_params=self._lgb_params,
            num_iterations=self.num_iterations,
            train_data=train_data,
            seed=self.seed,
        )
        self._forecaster = forecaster
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series for the forecast.

        Returns
        -------
        y_pred : pd.Series
            Point predictions.
        """
        rel = np.asarray(fh.to_relative(self.cutoff)._values, dtype=int)
        horizon = int(rel.max())
        times = self._train_len + np.arange(1, horizon + 1)
        test_data = pd.DataFrame(
            {
                "series_id": self._series_id,
                "date": self._dates[times - 1],
                "time": times,
            }
        )
        if self._x_cols:
            full = (
                ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
                .to_absolute(self.cutoff)
                .to_pandas()
            )
            X_full = X.reindex(full).ffill().bfill()
            for col in self._x_cols:
                test_data[col] = np.asarray(X_full[col].to_numpy())

        forecast = self._forecaster.forecast(test_data=test_data, type="forecast")

        values = forecast["fcst"].to_numpy()[rel - 1]
        index = fh.to_absolute(self.cutoff).to_pandas()
        return pd.Series(values, index=index, name=self._y.name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        lgb_params = {
            "learning_rate": 0.1,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "min_child_samples": 1,
        }
        params = [
            {
                "period": 4,
                "num_seasonal_components": 1,
                "num_iterations": 10,
                "lgb_params": lgb_params,
            },
            {
                "period": 4,
                "num_seasonal_components": 1,
                "variant": "paper",
                "num_iterations": 10,
                "lgb_params": lgb_params,
            },
        ]
        return params
