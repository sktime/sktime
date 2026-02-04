"""Dummy global forecaster for testing and baseline comparisons."""

__author__ = ["SimonBlanke"]
__all__ = ["DummyGlobalForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class DummyGlobalForecaster(BaseForecaster):
    """Dummy global forecaster that predicts mean of pretrain data.

    This forecaster implements pretraining by computing the mean
    of all time series in the pretrain set, then predicts that
    mean for all future time points.

    Useful for:

    - Testing the pretraining API
    - Baseline comparisons for global forecasting models
    - Educational examples

    Parameters
    ----------
    strategy : str, one of {"mean", "last", "mean_by_index"}, default="mean"
        Strategy for prediction:

        - "mean": predict mean of all values in pretrain set
        - "last": predict last value from fit data
        - "mean_by_index": predict mean computed per time index across pretrain series.
          Useful for cold start scenarios where pattern by index matters.

    Attributes
    ----------
    global_mean_ : float
        Mean of all values in pretrain set (set after pretrain)
    global_std_ : float
        Standard deviation across pretrain data (set after pretrain)
    n_pretrain_instances_ : int
        Number of instances in pretrain data (set after pretrain)
    n_pretrain_timepoints_ : int
        Total number of time points in pretrain data (set after pretrain)
    last_value_ : float or array-like
        Last value from fit data (set after fit)
    mean_by_index_ : pd.Series
        Mean value at each time index across pretrain series (set after pretrain
        when strategy="mean_by_index")

    Examples
    --------
    >>> from sktime.forecasting.dummy_global import DummyGlobalForecaster
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> # Create panel of training data
    >>> y_panel = _make_hierarchical(
    ...     hierarchy_levels=(2,), min_timepoints=10, max_timepoints=10
    ... )
    >>> forecaster = DummyGlobalForecaster()
    >>> forecaster.pretrain(y_panel)  # Learn global mean
    DummyGlobalForecaster()
    >>> # Now fit to a specific series
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster.fit(y)  # Set context
    DummyGlobalForecaster()
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # Predict global mean
    >>> y_pred.shape
    (3,)
    """

    _tags = {
        "capability:pretrain": True,
        "scitype:y": "both",
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "requires-fh-in-fit": False,
        "capability:pred_int": False,
        "capability:insample": False,
    }

    def __init__(self, strategy="mean"):
        self.strategy = strategy
        super().__init__()

    def _pretrain(self, y, X=None, fh=None):
        """Learn global statistics from panel data.

        Parameters
        ----------
        y : pd.DataFrame or pd.Series
            Panel data to learn from. If DataFrame with MultiIndex,
            should have (instance, time) levels.
        X : pd.DataFrame, optional
            Exogenous data (currently not used)
        fh : ForecastingHorizon, optional
            Forecasting horizon (currently not used)

        Returns
        -------
        self : reference to self
        """
        if isinstance(y, pd.Series):
            values = y.values
        elif isinstance(y, pd.DataFrame):
            values = y.values.flatten()
        else:
            values = np.asarray(y).flatten()

        self.global_mean_ = float(np.nanmean(values))
        self.global_std_ = float(np.nanstd(values))

        if isinstance(y, (pd.Series, pd.DataFrame)) and isinstance(
            y.index, pd.MultiIndex
        ):
            # Panel or hierarchical data with MultiIndex
            n_levels = y.index.nlevels
            if n_levels == 2:  # Panel data
                self.n_pretrain_instances_ = len(y.index.get_level_values(0).unique())
            else:
                # Hierarchical data: all levels except last are instance identifiers
                self.n_pretrain_instances_ = len(y.index.droplevel(-1).unique())
            self.n_pretrain_timepoints_ = len(y)

            if self.strategy == "mean_by_index":
                time_level = y.index.nlevels - 1
                if isinstance(y, pd.DataFrame):
                    self.mean_by_index_ = y.groupby(level=time_level).mean()
                else:
                    self.mean_by_index_ = y.groupby(level=time_level).mean()
        else:
            self.n_pretrain_instances_ = 1
            self.n_pretrain_timepoints_ = len(y)

            if self.strategy == "mean_by_index":
                if isinstance(y, pd.DataFrame):
                    self.mean_by_index_ = y.copy()
                else:
                    self.mean_by_index_ = y.copy()

        return self

    def _pretrain_update(self, y, X=None, fh=None):
        """Update global statistics with more data.

        This implements incremental learning by recomputing statistics.

        Parameters
        ----------
        y : pd.DataFrame or pd.Series
            Additional panel data to learn from
        X : pd.DataFrame, optional
            Exogenous data (currently not used)
        fh : ForecastingHorizon, optional
            Forecasting horizon (currently not used)

        Returns
        -------
        self : reference to self
        """
        return self._pretrain(y=y, X=X, fh=fh)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        For DummyGlobalForecaster, fit stores the last value
        and computes mean if not pretrained.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Time series to fit
        X : pd.DataFrame, optional
            Exogenous data (currently not used)
        fh : ForecastingHorizon, optional
            Forecasting horizon (currently not used)

        Returns
        -------
        self : reference to self
        """
        # Store last value for "last" strategy
        if isinstance(y, pd.DataFrame):
            self.last_value_ = y.iloc[-1].values
        else:
            self.last_value_ = y.iloc[-1]

        # If not pretrained, compute statistics from this series
        if not hasattr(self, "global_mean_"):
            if isinstance(y, pd.DataFrame):
                values = y.values.flatten()
            else:
                values = y.values
            self.global_mean_ = float(np.nanmean(values))

        if self.strategy == "mean_by_index" and not hasattr(self, "mean_by_index_"):
            if isinstance(y, pd.DataFrame):
                self.mean_by_index_ = y.copy()
            else:
                self.mean_by_index_ = y.copy()

        return self

    def _predict(self, fh, X=None):
        """Forecast using pretrained statistics or last value.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional
            Exogenous data (currently not used)

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
            Point predictions
        """
        fh_abs = fh.to_absolute_index(self.cutoff)
        if self.strategy == "mean_by_index":
            return self._predict_mean_by_index(fh_abs)

        # Determine prediction value based on strategy
        if self.strategy == "mean":
            pred_value = self.global_mean_
        elif self.strategy == "last":
            pred_value = self.last_value_
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                f"Must be one of ['mean', 'last', 'mean_by_index']"
            )

        # Check if we're dealing with multivariate data
        if isinstance(self._y, pd.DataFrame):
            n_cols = len(self._y.columns)
            if isinstance(pred_value, np.ndarray):
                # Repeat the last value for each time point
                data = np.tile(pred_value, (len(fh_abs), 1))
            else:
                # Single value - broadcast to all columns
                data = np.full((len(fh_abs), n_cols), pred_value)

            return pd.DataFrame(data, index=fh_abs, columns=self._y.columns)
        else:
            # Univariate
            if isinstance(pred_value, np.ndarray):
                pred_value = pred_value[0]
            return pd.Series(pred_value, index=fh_abs, name=self._y.name)

    def _predict_mean_by_index(self, fh_abs):
        """Predict using mean by index strategy.

        Maps forecast horizon indices to pretrain index means.
        Falls back to global_mean_ for indices not in pretrain data.

        Parameters
        ----------
        fh_abs : pd.Index
            Absolute forecast horizon index

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
            Point predictions
        """
        mean_idx = self.mean_by_index_.index

        # Check if we're dealing with multivariate data
        if isinstance(self._y, pd.DataFrame):
            n_cols = len(self._y.columns)
            data = np.full((len(fh_abs), n_cols), self.global_mean_)

            for i, idx in enumerate(fh_abs):
                if idx in mean_idx:
                    if isinstance(self.mean_by_index_, pd.DataFrame):
                        data[i, :] = self.mean_by_index_.loc[idx].values
                    else:
                        data[i, :] = self.mean_by_index_.loc[idx]

            return pd.DataFrame(data, index=fh_abs, columns=self._y.columns)
        else:
            # Univariate
            pred_values = []
            for idx in fh_abs:
                if idx in mean_idx:
                    pred_values.append(self.mean_by_index_.loc[idx])
                else:
                    pred_values.append(self.global_mean_)

            return pd.Series(pred_values, index=fh_abs, name=self._y.name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {"strategy": "mean"}
        params2 = {"strategy": "last"}
        params3 = {"strategy": "mean_by_index"}
        return [params1, params2, params3]
