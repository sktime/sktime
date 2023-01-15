# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for StatsForecast forecasters to be used in sktime framework."""

__author__ = ["FedericoGarza"]
__all__ = ["_StatsForecastAdapter"]


import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class _StatsForecastAdapter(BaseForecaster):
    """Base class for interfacing StatsForecast."""

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": True,  # does forecaster implement predict_quantiles?
        "python_dependencies": "statsforecast",
    }

    def __init__(self):
        self._forecaster = None
        super(_StatsForecastAdapter, self).__init__()

    def _instantiate_model(self):
        raise NotImplementedError("abstract method")

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self._forecaster = self._instantiate_model()
        self._forecaster.fit(y.values, X.values if X is not None else X)

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        # distinguish between in-sample and out-of-sample prediction
        fh_oos = fh.to_out_of_sample(self.cutoff)
        fh_ins = fh.to_in_sample(self.cutoff)

        # all values are out-of-sample
        if fh.is_all_out_of_sample(self.cutoff):
            return self._predict_fixed_cutoff(fh_oos, X=X)

        # all values are in-sample
        elif fh.is_all_in_sample(self.cutoff):
            return self._predict_in_sample(fh_ins, X=X)

        # both in-sample and out-of-sample values
        else:
            y_ins = self._predict_in_sample(fh_ins, X=X)
            y_oos = self._predict_fixed_cutoff(fh_oos, X=X)
            return pd.concat([y_ins, y_oos])

    def _predict_in_sample(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Generate in sample predictions.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """
        # initialize return objects
        fh_abs = fh.to_absolute(self.cutoff).to_numpy()
        fh_idx = fh.to_indexer(self.cutoff, from_cutoff=True)
        y_pred = pd.Series(index=fh_abs, dtype="float64")

        result = self._forecaster.predict_in_sample()
        y_pred.loc[fh_abs] = result["mean"].values[fh_idx]

        if return_pred_int:
            pred_ints = []
            for a in alpha:
                pred_int = pd.DataFrame(index=fh_abs, columns=["lower", "upper"])
                result = self._forecaster.predict_in_sample(level=int(100 * a))
                pred_int.loc[fh_abs] = result.drop("mean", axis=1).values[fh_idx, :]
                pred_ints.append(pred_int)
            return y_pred, pred_ints

        return y_pred

    def _predict_fixed_cutoff(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Make predictions out of sample.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).

        Returns
        -------
        y_pred : pandas.Series
        Returns series of predicted values.
        """
        n_periods = int(fh.to_relative(self.cutoff)[-1])
        result = self._forecaster.predict(
            h=n_periods,
            X=X.values if X is not None else X,
        )

        fh_abs = fh.to_absolute(self.cutoff)
        fh_idx = fh.to_indexer(self.cutoff)
        mean = pd.Series(result["mean"].values[fh_idx], index=fh_abs)
        if return_pred_int:
            pred_ints = []
            for a in alpha:
                result = self._forecaster.predict(
                    h=n_periods,
                    X=X.values if X is not None else X,
                    level=int(100 * a),
                )
                pred_int = result.drop("mean", axis=1).values
                pred_int = pd.DataFrame(
                    pred_int[fh_idx, :], index=fh_abs, columns=["lower", "upper"]
                )
                pred_ints.append(pred_int)
            return mean, pred_ints
        else:
            return pd.Series(mean, index=fh_abs)

    def _predict_interval(self, fh, X=None, coverage=0.90):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles
        State required:
            Requires state to be "fitted".
        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        # initializaing cutoff and fh related info
        cutoff = self.cutoff
        fh_oos = fh.to_out_of_sample(cutoff)
        fh_ins = fh.to_in_sample(cutoff)
        fh_is_in_sample = fh.is_all_in_sample(cutoff)
        fh_is_oosample = fh.is_all_out_of_sample(cutoff)

        # prepare the return DataFrame - empty with correct cols
        var_names = ["Coverage"]
        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(columns=int_idx)

        kwargs = {"X": X, "return_pred_int": True, "alpha": coverage}
        # all values are out-of-sample
        if fh_is_oosample:
            _, y_pred_int = self._predict_fixed_cutoff(fh_oos, **kwargs)

        # all values are in-sample
        elif fh_is_in_sample:
            _, y_pred_int = self._predict_in_sample(fh_ins, **kwargs)

        # if all in-sample/out-of-sample, we put y_pred_int in the required format
        if fh_is_in_sample or fh_is_oosample:
            # needs to be replaced, also seems duplicative, identical to part A
            for intervals, a in zip(y_pred_int, coverage):
                pred_int[("Coverage", a, "lower")] = intervals["lower"]
                pred_int[("Coverage", a, "upper")] = intervals["upper"]
            return pred_int

        # both in-sample and out-of-sample values (we reach this line only then)
        # in this case, we additionally need to concat in and out-of-sample returns
        _, y_ins_pred_int = self._predict_in_sample(fh_ins, **kwargs)
        _, y_oos_pred_int = self._predict_fixed_cutoff(fh_oos, **kwargs)
        for ins_int, oos_int, a in zip(y_ins_pred_int, y_oos_pred_int, coverage):
            pred_int[("Coverage", a, "lower")] = pd.concat([ins_int, oos_int])["lower"]
            pred_int[("Coverage", a, "upper")] = pd.concat([ins_int, oos_int])["upper"]

        return pred_int
