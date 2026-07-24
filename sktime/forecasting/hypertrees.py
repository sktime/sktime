# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hyper-Trees forecasters."""

__author__ = ["aminehd"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class HyperTreeNetARForecaster(BaseForecaster):
    """Hyper-TreeNet-AR forecaster, from the ``hypertrees-forecasting`` package.

    Direct interface to ``hypertrees.models.HyperTreeNetAR`` [1]_.

    Hyper-Trees use a gradient boosted tree (LightGBM) to learn the parameters
    of a classical time series model as functions of features, rather than
    forecasting the series directly. ``HyperTreeNetAR`` targets a time-varying
    AR(p) model: the tree produces embeddings that a small neural network maps
    to the AR parameters, and the AR model generates the forecast.

    The interfaced estimator is univariate and models a single series.

    Parameters
    ----------
    p : int, optional (default=2)
        Maximum number of AR(p) lags.
    embedding_dimension : int, optional (default=1)
        Embedding dimension of the tree embeddings fed to the network.
    hidden_dim : int, optional (default=128)
        Hidden dimension of the embedding network (MLP).
    dropout : float, optional (default=0.1)
        Dropout rate of the embedding network.
    use_random_projection : bool, optional (default=True)
        Whether to use random projections for the embeddings.
    rp_embed_dim : int, optional (default=12)
        Dimension of the random projections, only used when
        ``use_random_projection=True``.
    network_learning_rate : float, optional (default=1e-3)
        Learning rate of the embedding network optimizer.
    gradient_mode : str, optional (default="separate")
        Gradient computation mode, ``"separate"`` or ``"shared"``.
    device : str, optional (default="cpu")
        Device for the embedding network, e.g. ``"cpu"`` or ``"cuda"``.
    hessian_method : str, optional (default="exact")
        Method for the Hessian diagonal, ``"exact"`` or ``"gn"``.
    n_hessian_probes : int, optional (default=5)
        Number of Hutchinson probes, only used when ``hessian_method="gn"``.
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
    >>> from sktime.forecasting.hypertrees import HyperTreeNetARForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = HyperTreeNetARForecaster(p=2)  # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    HyperTreeNetARForecaster(...)
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    """

    _tags = {
        "authors": ["StatMixedML", "kashif", "aminehd"],
        "maintainers": ["aminehd"],
        "python_dependencies": ["hypertrees-forecasting>=0.2"],
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
        p=2,
        embedding_dimension=1,
        hidden_dim=128,
        dropout=0.1,
        use_random_projection=True,
        rp_embed_dim=12,
        network_learning_rate=1e-3,
        gradient_mode="separate",
        device="cpu",
        hessian_method="exact",
        n_hessian_probes=5,
        lgb_params=None,
        num_iterations=100,
        seed=123,
    ):
        self.p = p
        self.embedding_dimension = embedding_dimension
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_random_projection = use_random_projection
        self.rp_embed_dim = rp_embed_dim
        self.network_learning_rate = network_learning_rate
        self.gradient_mode = gradient_mode
        self.device = device
        self.hessian_method = hessian_method
        self.n_hessian_probes = n_hessian_probes
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
        from hypertrees.models.HyperTreeNetAR import HyperTreeNetAR

        self._series_id = 0
        self._train_len = len(y)
        freq = getattr(y.index, "freqstr", None)
        if freq is None and isinstance(y.index, pd.DatetimeIndex):
            try:
                freq = pd.infer_freq(y.index)
            except ValueError:
                freq = None
        self._freq = freq or "MS"
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
                train_data[str(col)] = np.asarray(X[col].to_numpy())

        forecaster = HyperTreeNetAR(
            p=self.p,
            freq=self._freq,
            fcst_h=fcst_h,
            device=self.device,
            hessian_method=self.hessian_method,
            n_hessian_probes=self.n_hessian_probes,
        )
        network_params = {
            "learning_rate": self.network_learning_rate,
            "embedding_dimension": self.embedding_dimension,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "use_random_projection": self.use_random_projection,
            "rp_embed_dim": self.rp_embed_dim,
        }
        forecaster.train(
            lgb_params=self._lgb_params,
            network_params=network_params,
            gradient_mode=self.gradient_mode,
            num_iterations=self.num_iterations,
            train_data=train_data,
            seed=self.seed,
        )
        forecaster.dataset_references = None
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
                test_data[str(col)] = np.asarray(X_full[col].to_numpy())

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
                "p": 2,
                "num_iterations": 10,
                "lgb_params": lgb_params,
                "hidden_dim": 8,
                "rp_embed_dim": 4,
                "dropout": 0.0,
            },
            {
                "p": 1,
                "gradient_mode": "shared",
                "num_iterations": 10,
                "lgb_params": lgb_params,
                "hidden_dim": 8,
                "rp_embed_dim": 4,
                "dropout": 0.0,
            },
        ]
        return params
