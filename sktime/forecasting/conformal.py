# -*- coding: utf-8 -*-
"""Implements simple conformal forecast intervals.

Code based partially on NaiveVariance by ilyasmoutawwakil.
"""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["ConformalIntervals"]
__author__ = ["fkiraly", "bethrice44"]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.datatypes import convert, convert_to
from sktime.datatypes._utilities import get_slice
from sktime.forecasting.base import BaseForecaster


class ConformalIntervals(BaseForecaster):
    r"""Empirical and conformal prediction intervals.

    Implements empirical and conformal prediction intervals, on absolute residuals.
    Empirical prediction intervals are based on sliding window empirical quantiles.
    Conformal prediction intervals are implemented as described in [1]_.

    All intervals wrap an arbitrary forecaster, i.e., add probabilistic prediction
    capability to a given point prediction forecaster (first argument).

    method="conformal_bonferroni" is the method described in [1]_,
        where an arbitrary forecaster is used instead of the RNN.
    method="conformal" is the method in [1]_, but without Bonferroni correction.
        i.e., separate forecasts are made which results in H=1 (at all horizons).
    method="empirical" uses quantiles of relative signed residuals on training set,
        i.e., y_t+h^(i) - y-hat_t+h^(i), ranging over i, in the notation of [1]_,
        at quantiles 0.5-0.5*coverage (lower) and 0.5+0.5*coverage (upper),
        as offsets to the point prediction at forecast horizon h
    method="empirical_residual" uses empirical quantiles of absolute residuals
        on the training set, i.e., quantiles of epsilon-h (in notation [1]_),
        at quantile point (1-coverage)/2 quantiles, as offsets to point prediction

    Parameters
    ----------
    forecaster : estimator
        Estimator to which probabilistic forecasts are being added
    method : str, optional, default="empirical"
        "empirical": predictive interval bounds are empirical quantiles from training
        "empirical_residual": upper/lower are plusminus (1-coverage)/2 quantiles
            of the absolute residuals at horizon, i.e., of epsilon-h
        "conformal_bonferroni": Bonferroni, as in Stankeviciute et al
            Caveat: this does not give frequentist but conformal predictive intervals
        "conformal": as in Stankeviciute et al, but with H=1,
            i.e., no Bonferroni correction under number of indices in the horizon
    initial_window : int, optional, default=1
        number of minimum initial indices to use for fitting when computing residuals
    sample_frac : float, optional, default=None
        value in range (0,1) corresponding to fraction of y index to calculate
        residuals matrix values for (for speeding up calculation)
    verbose : bool, optional, default=False
        whether to print warnings if windows with too few data points occur

    References
    ----------
    .. [1] Kamile Stankeviciute, Ahmed M Alaa and Mihaela van der Schaar.
        Conformal Time Series Forecasting. NeurIPS 2021.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.conformal import ConformalIntervals
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> conformal_forecaster = ConformalIntervals(forecaster)
    >>> conformal_forecaster.fit(y, fh=[1,2,3])
    ConformalIntervals(...)
    >>> pred_int = conformal_forecaster.predict_interval()
    """

    _required_parameters = ["forecaster"]
    _tags = {
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
    }

    ALLOWED_METHODS = [
        "empirical",
        "empirical_residual",
        "conformal",
        "conformal_bonferroni",
    ]

    def __init__(
        self,
        forecaster,
        method="empirical",
        initial_window=1,
        sample_frac=None,
        verbose=False,
    ):

        if not isinstance(method, str):
            raise TypeError(f"method must be a str, one of {self.ALLOWED_METHODS}")

        if method not in self.ALLOWED_METHODS:
            raise ValueError(
                f"method must be one of {self.ALLOWED_METHODS}, but found {method}"
            )

        self.forecaster = forecaster
        self.method = method
        self.verbose = verbose
        self.initial_window = initial_window
        self.sample_frac = sample_frac

        super(ConformalIntervals, self).__init__()

        tags_to_clone = [
            "requires-fh-in-fit",
            "ignores-exogeneous-X",
            "handles-missing-data",
            "y_inner_mtype",
            "X_inner_mtype",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(self.forecaster, tags_to_clone)

    def _fit(self, y, X=None, fh=None):
        self.fh_early_ = fh is not None
        self.forecaster_ = clone(self.forecaster)
        self.forecaster_.fit(y=y, X=X, fh=fh)

        if self.fh_early_:
            self.residuals_matrix_ = self._compute_sliding_residuals(
                y=y,
                X=X,
                forecaster=self.forecaster,
                initial_window=self.initial_window,
                sample_frac=self.sample_frac,
            )

        return self

    def _predict(self, fh, X=None):
        return self.forecaster_.predict(fh=fh, X=X)

    def _update(self, y, X=None, update_params=True):
        self.forecaster_.update(y, X, update_params=update_params)
        return self

    def _predict_interval(self, fh, X=None, coverage=None):
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
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        fh_relative = fh.to_relative(self.cutoff)
        fh_absolute = fh.to_absolute(self.cutoff)

        if self.fh_early_:
            residuals_matrix = self.residuals_matrix_
        else:
            residuals_matrix = self._compute_sliding_residuals(
                y=self._y,
                X=self._X,
                forecaster=self.forecaster,
                initial_window=self.initial_window,
                sample_frac=self.sample_frac,
            )

        ABS_RESIDUAL_BASED = ["conformal", "conformal_bonferroni", "empirical_residual"]

        cols = pd.MultiIndex.from_product([["Coverage"], coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(index=fh_absolute, columns=cols)
        for fh_ind, offset in zip(fh_absolute, fh_relative):
            resids = np.diagonal(residuals_matrix, offset=offset)
            resids = resids[~np.isnan(resids)]
            abs_resids = np.abs(resids)
            coverage2 = np.repeat(coverage, 2)
            if self.method == "empirical":
                quantiles = 0.5 + np.tile([-0.5, 0.5], len(coverage)) * coverage2
                pred_int_row = np.quantile(resids, quantiles)
            if self.method == "empirical_residual":
                quantiles = 0.5 - 0.5 * coverage2
                pred_int_row = np.quantile(abs_resids, quantiles)
            elif self.method == "conformal_bonferroni":
                alphas = 1 - coverage2
                quantiles = 1 - alphas / len(fh)
                pred_int_row = np.quantile(abs_resids, quantiles)
            elif self.method == "conformal":
                quantiles = coverage2
                pred_int_row = np.quantile(abs_resids, quantiles)

            pred_int.loc[fh_ind] = pred_int_row

        y_pred = self.predict(fh=fh, X=X)
        y_pred = convert(y_pred, from_type=self._y_mtype_last_seen, to_type="pd.Series")
        y_pred.index = fh_absolute

        for col in cols:
            if self.method in ABS_RESIDUAL_BASED:
                sign = 1 - 2 * (col[2] == "lower")
            else:
                sign = 1
            pred_int[col] = y_pred + sign * pred_int[col]

        return pred_int.convert_dtypes()

    def _compute_sliding_residuals(self, y, X, forecaster, initial_window, sample_frac):
        """Compute sliding residuals used in uncertainty estimates.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            sktime compatible time series to use in computing residuals matrix
        X : pd.DataFrame
            sktime compatible exogeneous time series to use in forecasts
        forecaster : sktime compatible forecaster
            forecaster to use in computing the sliding residuals
        initial_window : int
            minimum length of initial window to use in fitting
        sample_frac : float
            for speeding up computing of residuals matrix.
            sample value in range (0, 1) to obtain a fraction of y indices to
            compute residuals matrix for
        Returns
        -------
        residuals_matrix : pd.DataFrame, row and column index = y.index[initial_window:]
            [i,j]-th entry is signed residual of forecasting y.loc[j] from y.loc[:i],
            using a clone of the forecaster passed through the forecaster arg.
            if sample_frac is passed this will have NaN values for 1 - sample_frac
            fraction of the matrix
        """
        y = convert_to(y, "pd.Series")

        y_index = y.index[initial_window:]

        residuals_matrix = pd.DataFrame(columns=y_index, index=y_index, dtype="float")

        if sample_frac:
            y_index = y_index.to_series().sample(frac=sample_frac)

        for id in y_index:
            forecaster = clone(forecaster)
            y_train = get_slice(y, start=None, end=id)  # subset on which we fit
            y_test = get_slice(y, start=id, end=None)  # subset on which we predict

            X_train = get_slice(X, start=None, end=id)
            X_test = get_slice(X, start=id, end=None)

            try:
                forecaster.fit(y_train, X=X_train, fh=y_test.index)
            except ValueError:
                warn(
                    f"Couldn't fit the model on "
                    f"time series window length {len(y_train)}.\n"
                )
                continue
            try:
                residuals_matrix.loc[id] = forecaster.predict_residuals(y_test, X_test)
            except IndexError:
                warn(
                    f"Couldn't predict after fitting on time series of length \
                     {len(y_train)}.\n"
                )

        return residuals_matrix

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        params_list = {"forecaster": FORECASTER}

        return params_list
