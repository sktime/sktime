#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements NeuralProphet forecaster by wrapping neuralprophet."""

__author__ = ["vedantag17"]
__all__ = ["NeuralProphet"]

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class NeuralProphet(BaseForecaster):
    """NeuralProphet forecaster by wrapping NeuralProphet algorithm [1]_.

    Direct interface to NeuralProphet, using the sktime interface.
    All hyper-parameters are exposed via the constructor.

    Data can be passed in one of the sktime compatible formats.
    Like Prophet, NeuralProphet also supports integer/range and period index:
    * integer/range index is interpreted as days since Jan 1, 2000
    * ``PeriodIndex`` is converted using the ``pandas`` method ``to_timestamp``

    Parameters
    ----------
    freq : str, optional
        Frequency of time series (e.g. 'D', 'M', etc.)
    add_seasonality : dict, optional
        Additional seasonality component parameters
    custom_seasonalities : list of dict, optional
        Custom seasonality components
    add_country_holidays : dict, optional
        Country holidays to include
    growth : str, default="linear"
        Type of trend ('linear' or 'flat')
    changepoints : list, optional
        List of dates for trend changepoints
    n_changepoints : int, default=10
        Number of potential trend changepoints
    changepoints_range : float, default=0.8
        Proportion of history for changepoints
    yearly_seasonality : bool, default=True
        Whether to include yearly seasonality
    weekly_seasonality : bool, default=True
        Whether to include weekly seasonality
    daily_seasonality : bool, default=False
        Whether to include daily seasonality
    seasonality_mode : str, default="additive"
        How seasonality is combined ('additive' or 'multiplicative')
    seasonality_reg : float, default=0
        Regularization strength for seasonality
    holidays : pd.DataFrame, optional
        Custom holidays DataFrame
    holidays_mode : str, default="additive"
        How holidays are combined ('additive' or 'multiplicative')
    holidays_reg : float, default=0
        Regularization strength for holidays
    trend_reg : float, default=0
        Regularization strength for trend
    trend_reg_threshold : bool, default=False
        Threshold for trend regularization
    learning_rate : float, default=None
        Maximum learning rate (applicable in quasi-Newton optimization)
    epochs : int, default=None
        Number of training epochs
    batch_size : int, default=None
        Number of samples per mini-batch
    loss_func : str, default="Huber"
        Type of loss to use (e.g., "Huber", "MSE", "MAE", etc.)
    alpha : float, default=0.05
        Width of the uncertainty intervals
    uncertainty_samples : int, default=1000
        Number of samples for estimating uncertainty intervals
    verbose : bool, default=False
        Whether to print status information during fitting

    References
    ----------
    .. [1] https://github.com/ourownstory/neural_prophet

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.neuralprophet import NeuralProphet
    >>> # NeuralProphet requires data with a pandas.DatetimeIndex
    >>> y = load_airline().to_timestamp(freq='M')
    >>> forecaster = NeuralProphet(  # doctest: +SKIP
    ...     n_changepoints=0,
    ...     yearly_seasonality=False,
    ...     weekly_seasonality=False,
    ...     daily_seasonality=False,
    ...     epochs=5,
    ...     uncertainty_samples=0,
    ...     verbose=False
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    NeuralProphet(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3]) #doctest: +SKIP
    """

    _tags = {
        "authors": ["vedantag17"],
        "maintainers": ["vedantag17"],
        "python_dependencies": ["neuralprophet", "setuptools"],
        # neuralprophet causes a C-level segfault on Windows + Python 3.13+
        "env_marker": 'platform_system != "Windows" or python_version < "3.13"',
        "capability:exogenous": True,
        "capability:missing_values": True,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "tests:vm": True,
        "tests:skip_by_name": ["test_fit_idempotent"],
    }

    def __init__(
        self,
        freq=None,
        add_seasonality=None,
        custom_seasonalities=None,
        add_country_holidays=None,
        growth="linear",
        changepoints=None,
        n_changepoints=10,
        changepoints_range=0.8,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        seasonality_reg=0,
        holidays=None,
        holidays_mode="additive",
        holidays_reg=0,
        trend_reg=0,
        trend_reg_threshold=False,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="Huber",
        alpha=DEFAULT_ALPHA,
        uncertainty_samples=1000,
        verbose=False,
    ):
        self.freq = freq
        self.add_seasonality = add_seasonality
        self.custom_seasonalities = custom_seasonalities
        self.add_country_holidays = add_country_holidays
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoints_range = changepoints_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_reg = seasonality_reg
        self.holidays = holidays
        self.holidays_mode = holidays_mode
        self.holidays_reg = holidays_reg
        self.trend_reg = trend_reg
        self.trend_reg_threshold = trend_reg_threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.alpha = alpha
        self.uncertainty_samples = uncertainty_samples
        self.verbose = verbose

        super().__init__()

    @staticmethod
    def _neuralprophet_weights_only_patch():
        """Context manager: force ``weights_only=False`` during NeuralProphet fit.

        PyTorch >= 2.6 defaults ``weights_only`` to ``True`` (and treats
        ``None`` as ``True``).  NeuralProphet's PyTorch Lightning training
        loop (LR finder, model checkpointing, etc.) serialises arbitrary
        ``nn.Module`` subclasses into its checkpoint file.  These cannot be
        fully enumerated in a safe-globals allowlist, so we patch
        ``torch.load`` itself to strip ``weights_only`` for the duration of
        NeuralProphet's own fit / pickle roundtrip.

        The patch is active only inside the ``with`` block, so it does not
        affect any other ``torch.load`` calls in the process.
        """
        import contextlib

        import torch

        _original_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            # Force weights_only=False so NeuralProphet's Lightning checkpoint
            # (which contains arbitrary nn.Module subclasses) can be loaded.
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        @contextlib.contextmanager
        def _ctx():
            torch.load = _patched_torch_load
            try:
                yield
            finally:
                torch.load = _original_torch_load

        return _ctx()

    def __deepcopy__(self, memo):
        """Handle deepcopy by copy-serializing the non-deepcopy-able PyTorch model.

        NeuralProphet's underlying PyTorch model uses ``weight_norm`` internally,
        which makes tensors non-deepcopy-able (see pytorch/pytorch#103001).

        PyTorch 2.6 changed the default of ``torch.load`` to ``weights_only=True``.
        Lightning's checkpoint restore (used internally during NeuralProphet's fit)
        passes ``weights_only=True`` explicitly and the checkpoint contains
        arbitrary ``nn.Module`` subclasses that cannot all be allowlisted.
        We patch lightning_fabric's ``_load`` for the duration of the pickle
        roundtrip to avoid the restriction.
        """
        import copy
        import pickle

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_model":
                try:
                    with self._neuralprophet_weights_only_patch():
                        setattr(result, k, pickle.loads(pickle.dumps(v)))
                except Exception:
                    # Fallback to direct reference if pickling fails entirely.
                    setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        import pandas as pd
        from neuralprophet import NeuralProphet as _NeuralProphet

        # Convert PeriodIndex to DatetimeIndex if needed
        if hasattr(y.index, "to_timestamp"):
            y = y.copy()
            y.index = y.index.to_timestamp()

        # Prepare data for NeuralProphet
        df = pd.DataFrame({"ds": y.index, "y": y.values})

        # Initialize and configure NeuralProphet model
        self._model = _NeuralProphet(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoints_range=self.changepoints_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_reg=self.seasonality_reg,
            trend_reg=self.trend_reg,
            trend_reg_threshold=self.trend_reg_threshold,
            learning_rate=self.learning_rate,
            n_forecasts=1,
            epochs=self.epochs,
            batch_size=self.batch_size,
            loss_func=self.loss_func,
            quantiles=[self.alpha / 2, 1 - self.alpha / 2]
            if self.uncertainty_samples
            else [],
        )

        # Add regressors
        self._regressors = []
        if X is not None:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, index=y.index)
            else:
                X = X.copy()
                X = X.reindex(y.index)

            # Coerce integer column names to strings
            X.columns = [
                f"regressor_{c}" if not isinstance(c, str) else c for c in X.columns
            ]

            for col in X.columns:
                if X[col].notna().any():
                    self._model.add_future_regressor(col)
                    self._regressors.append(col)

            if self._regressors:
                df = df.join(X[self._regressors].reset_index(drop=True), how="left")

        # Store training dataframe for future predictions
        self._training_df = df.copy()

        # Add custom seasonalities, holidays, etc.
        if self.custom_seasonalities:
            for seasonality in self.custom_seasonalities:
                self._model.add_seasonality(**seasonality)

        if self.add_country_holidays:
            self._model.add_country_holidays(**self.add_country_holidays)

        if self.holidays is not None:
            for holiday in self.holidays.itertuples():
                self._model.add_events(
                    holiday.holiday,
                    mode=self.holidays_mode,
                    regularization=self.holidays_reg,
                )

        # Fit the model — patch lightning_fabric for the duration so that
        # NeuralProphet's internal torch.load (LR finder / checkpoint restore)
        # does not fail under PyTorch >= 2.6 weights_only=True restrictions.
        with self._neuralprophet_weights_only_patch():
            self._model.fit(df)

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        import numpy as np
        import pandas as pd

        absolute_fh = self.fh.to_absolute(self.cutoff)
        fh_index = absolute_fh.to_pandas()

        # Convert fh_index to the same dtype as _training_df["ds"] so that
        # set-membership checks and concat don't produce mixed-type columns.
        if pd.api.types.is_datetime64_any_dtype(self._training_df["ds"]):
            if hasattr(fh_index, "to_timestamp"):
                fh_ds = list(fh_index.to_timestamp())
            else:
                fh_ds = list(pd.DatetimeIndex(fh_index))
        else:
            fh_ds = list(fh_index)

        # Coerce X column names once so regressor lookups are consistent
        if X is not None and hasattr(self, "_regressors") and self._regressors:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X = X.copy()
            X.columns = [
                f"regressor_{c}" if not isinstance(c, str) else c for c in X.columns
            ]
        else:
            X = None

        # Split fh into in-sample (already in training_df) and out-of-sample.
        # Only out-of-sample rows are appended; in-sample rows already exist and
        # must NOT be duplicated — NeuralProphet rejects duplicate ds values.
        training_ds_set = set(self._training_df["ds"].tolist())
        oos_mask = [ds not in training_ds_set for ds in fh_ds]
        oos_ds = [ds for ds, oos in zip(fh_ds, oos_mask) if oos]
        oos_indices = [i for i, oos in enumerate(oos_mask) if oos]

        if oos_ds:
            n_oos = len(oos_ds)
            future_row_data = {"ds": oos_ds, "y": np.full(n_oos, np.nan)}
            if X is not None:
                for col in self._regressors:
                    if col in X.columns:
                        future_row_data[col] = X[col].values[oos_indices]
                    else:
                        future_row_data[col] = np.full(n_oos, np.nan)
            combined_df = pd.concat(
                [self._training_df, pd.DataFrame(future_row_data)],
                ignore_index=True,
            )
        else:
            combined_df = self._training_df.copy()

        forecast = self._model.predict(combined_df)

        # Extract yhat1 at each requested ds position (handles mixed in/out-of-sample)
        forecast_ds = forecast["ds"].tolist()
        ds_to_yhat = dict(zip(forecast_ds, forecast["yhat1"].tolist()))
        yhat = [ds_to_yhat.get(ds, np.nan) for ds in fh_ds]

        return pd.Series(yhat, index=fh_index, name=self._y.name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params1 = {
            "n_changepoints": 0,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "epochs": 5,
            "uncertainty_samples": 0,
            "verbose": False,
        }
        params2 = {
            "growth": "linear",
            "n_changepoints": 0,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "epochs": 5,
            "uncertainty_samples": 0,
            "verbose": False,
        }
        params3 = {
            "n_changepoints": 0,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "epochs": 5,
            "uncertainty_samples": 10,
            "alpha": 0.1,
            "verbose": False,
        }
        return [params1, params2, params3]
