# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Base class template for forecaster scitype.

    class name: BaseForecaster

Scitype defining methods:
    fitting            - fit(y, X=None, fh=None)
    forecasting        - predict(fh=None, X=None)
    updating           - update(y, X=None, update_params=True)

Convenience methods:
    fit&forecast       - fit_predict(y, X=None, fh=None)
    update&forecast    - update_predict(cv=None, X=None, update_params=True)
    forecast residuals - predict_residuals(y, X=None, fh=None)
    forecast scores    - score(y, X=None, fh=None)

Optional, special capability methods (check capability tags if available):
    forecast intervals - predict_interval(fh=None, X=None, coverage=0.90)
    forecast quantiles - predict_quantiles(fh=None, X=None, alpha=[0.05, 0.95])

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()
    current ForecastingHorizon  - fh

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__author__ = ["mloning", "big-o", "fkiraly", "sveameyer13"]

__all__ = ["BaseForecaster"]

from contextlib import contextmanager
from warnings import warn

import numpy as np
import pandas as pd
from sklearn import clone

from sktime.base import BaseEstimator
from sktime.datatypes import (
    VectorizedDF,
    check_is_scitype,
    convert_to,
    get_cutoff,
    mtype_to_scitype,
)
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.datetime import _shift
from sktime.utils.validation._dependencies import _check_dl_dependencies
from sktime.utils.validation.forecasting import check_alpha, check_cv, check_fh, check_X
from sktime.utils.validation.series import check_equal_time_index

DEFAULT_ALPHA = 0.05


def _coerce_to_list(obj):
    """Return [obj] if obj is not a list, otherwise obj."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class BaseForecaster(BaseEstimator):
    """Base forecaster template class.

    The base forecaster specifies the methods and method
    signatures that all forecasters have to implement.

    Specific implementations of these methods is deferred to concrete
    forecasters.
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": True,  # does estimator ignore the exogeneous X?
        "capability:pred_int": False,  # can the estimator produce prediction intervals?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict, support for X?
        "requires-fh-in-fit": True,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
    }

    def __init__(self):
        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        self._converter_store_y = dict()  # storage dictionary for in/output conversion

        super(BaseForecaster, self).__init__()

    def fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets self.cutoff and self._cutoff to last index seen in `y`.
            Sets fitted model attributes ending in "_".
            Stores fh to self.fh if fh is passed.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
            if self.get_tag("requires-fh-in-fit"), must be passed, not optional
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index

        Returns
        -------
        self : Reference to self.
        """
        # check y is not None
        assert y is not None, "y cannot be None, but found None"

        # if fit is called, fitted state is re-set
        self._is_fitted = False

        fh = self._check_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X/y to the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        #####################################################
        vectorization_needed = isinstance(y_inner, VectorizedDF)
        self._is_vectorized = vectorization_needed
        # we call the ordinary _fit if no looping/vectorization needed
        if not vectorization_needed:
            self._fit(y=y_inner, X=X_inner, fh=fh)
        else:
            # otherwise we call the vectorized version of fit
            self._vectorize("fit", y=y_inner, X=X_inner, fh=fh)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(
        self,
        fh=None,
        X=None,
    ):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores fh to self.fh if fh is passed and has not been passed previously.

        Parameters
        ----------
        fh : int, list, np.ndarray or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, or 2D np.ndarray, optional (default=None)
            Exogeneous time series to predict from
            if self.get_tag("X-y-must-have-same-index"), X.index must contain fh.index

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y passed in fit (most recently)
        """
        # handle inputs

        self.check_is_fitted()
        fh = self._check_fh(fh)

        # input check and conversion for X
        X_inner = self._check_X(X=X)

        # we call the ordinary _predict if no looping/vectorization needed
        if not self._is_vectorized:
            y_pred = self._predict(fh=fh, X=X_inner)
        else:
            # otherwise we call the vectorized version of predict
            y_pred = self._vectorize("predict", X=X_inner, fh=fh)

        # convert to output mtype, identical with last y mtype seen
        y_out = convert_to(
            y_pred,
            self._y_mtype_last_seen,
            store=self._converter_store_y,
            store_behaviour="freeze",
        )

        return y_out

    def fit_predict(self, y, X=None, fh=None):
        """Fit and forecast time series at future horizon.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets self.cutoff and self._cutoff to last index seen in `y`.
            Sets fitted model attributes ending in "_".
            Stores fh to self.fh.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon (not optional)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to and to predict from
            if self.get_tag("X-y-must-have-same-index"),
            X.index must contain y.index and fh.index

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        fh = self._check_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X/y to the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # apply fit and then predict
        vectorization_needed = isinstance(y_inner, VectorizedDF)
        self._is_vectorized = vectorization_needed
        # we call the ordinary _fit if no looping/vectorization needed
        if not vectorization_needed:
            self._fit(y=y_inner, X=X_inner, fh=fh)
        else:
            # otherwise we call the vectorized version of fit
            self._vectorize("fit", y=y_inner, X=X_inner, fh=fh)

        self._is_fitted = True
        # call the public predict to avoid duplicating output conversions
        #  input conversions are skipped since we are using X_inner
        return self.predict(fh=fh, X=X_inner)

    def predict_quantiles(self, fh=None, X=None, alpha=None):
        """Compute/return quantile forecasts.

        If alpha is iterable, multiple quantiles will be calculated.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores fh to self.fh if fh is passed and has not been passed previously.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : float or list of float, optional (default=[0.05, 0.95])
            A probability or list of, at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "quantile predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()
        # input checks
        if alpha is None:
            alpha = [0.05, 0.95]
        fh = self._check_fh(fh)

        alpha = check_alpha(alpha)

        # input check and conversion for X
        X_inner = self._check_X(X=X)

        quantiles = self._predict_quantiles(fh=fh, X=X_inner, alpha=alpha)
        return quantiles

    def predict_interval(
        self,
        fh=None,
        X=None,
        coverage=0.90,
    ):
        """Compute/return prediction interval forecasts.

        If coverage is iterable, multiple intervals will be calculated.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores fh to self.fh if fh is passed and has not been passed previously.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        coverage : float or list of float, optional (default=0.90)
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
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "prediction intervals. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()
        # input checks
        fh = self._check_fh(fh)
        coverage = check_alpha(coverage)

        # check and convert X
        X_inner = self._check_X(X=X)

        pred_int = self._predict_interval(fh=fh, X=X_inner, coverage=coverage)

        # todo: remove if changing pred_interval format
        # if pred_int.columns.nlevels == 3:
        #     pred_int = _convert_pred_interval_to_quantiles(pred_int)

        return pred_int

    def predict_var(
        self,
        fh=None,
        X=None,
    ):
        """Compute/return variance forecasts.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores fh to self.fh if fh is passed and has not been passed previously.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        pred_var : pd.DataFrame
            Column names are exactly those of `y` passed in `fit`/`update`.
                For nameless formats, column index will be a RangeIndex.
            Row index is fh. Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "variance predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()
        # input checks
        fh = self._check_fh(fh)

        # check and convert X
        X_inner = self._check_X(X=X)

        pred_var = self._predict_var(fh=fh, X=X_inner)

        return pred_var

    def predict_proba(
        self,
        fh=None,
        X=None,
    ):
        """Compute/return fully probabilistic forecasts.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores fh to self.fh if fh is passed and has not been passed previously.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        pred_dist : tfp Distribution object
            batch shape is 1D and same length as fh
            event shape is 1D, with length equal to number of variables being forecast
            i-th (batch) distribution is forecast for i-th entry of fh
            j-th (event) component is j-th variable, same order as y in `fit`/`update`
        """
        msg = (
            "tensorflow-probability must be installed for fully probabilistic forecasts"
            "install `sktime` deep learning dependencies by `pip install sktime[dl]`"
        )
        _check_dl_dependencies(msg)

        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "fully probabilistic predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()
        # input checks
        fh = self._check_fh(fh)

        # check and convert X
        X_inner = self._check_X(X=X)

        pred_dist = self._predict_proba(fh=fh, X=X_inner)

        return pred_dist

    def update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        If no estimator-specific update method has been implemented,
        default fall-back is as follows:
            update_params=True: fitting to all observed data so far
            update_params=False: updates cutoff and remembers data only

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self.cutoff, self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with `y` and `X`, by appending rows.
            Updates self. cutoff and self._cutoff to last index seen in `y`.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, or 2D np.ndarray optional (default=None)
            Exogeneous time series to fit to
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        self.check_is_fitted()

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update internal X/y with the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        self._update(y=y_inner, X=X_inner, update_params=update_params)

        return self

    def update_predict(
        self,
        y,
        cv=None,
        X=None,
        update_params=True,
    ):
        """Make predictions and update model iteratively over the test set.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self.cutoff, self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with `y` and `X`, by appending rows.
            Updates self.cutoff and self._cutoff to last index seen in `y`.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        cv : temporal cross-validation generator, optional (default=None)
        X : pd.DataFrame, or 2D np.ndarray optional (default=None)
            Exogeneous time series to fit to and predict from
            if self.get_tag("X-y-must-have-same-index"),
            X.index must contain y.index and fh.index
        update_params : bool, optional (default=True)

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y
        """
        self.check_is_fitted()

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        cv = check_cv(cv)

        return self._predict_moving_cutoff(
            y=y_inner,
            cv=cv,
            X=X_inner,
            update_params=update_params,
        )

    def update_predict_single(
        self,
        y=None,
        fh=None,
        X=None,
        update_params=True,
    ):
        """Update model with new data and make forecasts.

        This method is useful for updating and making forecasts in a single step.

        If no estimator-specific update method has been implemented,
        default fall-back is first update, then predict.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self.cutoff, self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with `y` and `X`, by appending rows.
            Updates self. cutoff and self._cutoff to last index seen in `y`.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Target time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to and to predict from
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain y.index and fh.index
        update_params : bool, optional (default=False)

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y
        """
        self.check_is_fitted()
        fh = self._check_fh(fh)

        if y is None:
            raise ValueError("y must be of Series type and cannot be None")

        self.check_is_fitted()
        fh = self._check_fh(fh)

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update internal _X/_y with the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        return self._update_predict_single(
            y=y_inner,
            fh=fh,
            X=X_inner,
            update_params=update_params,
        )

    def predict_residuals(self, y=None, X=None):
        """Return residuals of time series forecasts.

        Residuals will be computed for forecasts at y.index.

        If fh must be passed in fit, must agree with y.index.
        If y is an np.ndarray, and no fh has been passed in fit,
        the residuals will be computed at a fh of range(len(y.shape[0]))

        State required:
            Requires state to be "fitted".
            If fh has been set, must correspond to index of y (pandas or integer)

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores y.index to self.fh if has not been passed previously.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, np.ndarray (1D or 2D), or None
            Time series with ground truth observations, to compute residuals to.
            Must have same type, dimension, and indices as expected return of predict.
            if None, the y seen so far (self._y) are used, in particular:
                if preceded by a single fit call, then in-sample residuals are produced
                if fit requires fh, it must have pointed to index of y in fit
        X : pd.DataFrame, or 2D np.ndarray, optional (default=None)
            Exogeneous time series to predict from
            if self.get_tag("X-y-must-have-same-index"), X.index must contain fh.index

        Returns
        -------
        y_res : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Forecast residuals at fh, with same index as fh
            y_pred has same type as y passed in fit (most recently)
        """
        # if no y is passed, the so far observed y is used
        if y is None:
            y = self._y

        # we want residuals, so fh must be the index of y
        # if data frame: take directly from y
        # to avoid issues with _set_fh, we convert to relative if self.fh is
        if isinstance(y, (pd.DataFrame, pd.Series)):
            fh = ForecastingHorizon(y.index, is_relative=False)
            if self._fh is not None and self.fh.is_relative:
                fh = fh.to_relative(self.cutoff)
            fh = self._check_fh(fh)
        # if np.ndarray, rows are not indexed
        # so will be interpreted as range(len), or existing fh if it is stored
        elif isinstance(y, np.ndarray):
            if self._fh is None:
                fh = range(y.shape[0])
            else:
                fh = self.fh
        else:
            raise TypeError("y must be a supported Series mtype")

        y_pred = self.predict(fh=fh, X=X)

        if not type(y_pred) == type(y):
            raise TypeError(
                "y must have same type, dims, index as expected predict return. "
                f"expected type {type(y_pred)}, but found {type(y)}"
            )

        y_res = y - y_pred

        return y_res

    def score(self, y, X=None, fh=None):
        """Scores forecast against ground truth, using MAPE.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to score
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, array-like or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to score
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index

        Returns
        -------
        score : float
            sMAPE loss of self.predict(fh, X) with respect to y_test.

        See Also
        --------
        :meth:`sktime.performance_metrics.forecasting.mean_absolute_percentage_error`
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        # symmetric=True is default for mean_absolute_percentage_error
        from sktime.performance_metrics.forecasting import (
            mean_absolute_percentage_error,
        )

        return mean_absolute_percentage_error(y, self.predict(fh, X))

    def get_fitted_params(self):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        raise NotImplementedError("abstract method")

    def _check_X_y(self, X=None, y=None):
        """Check and coerce X/y for fit/predict/update functions.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D), optional (default=None)
            Time series to check.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series.

        Returns
        -------
        y_inner : Series, Panel, or Hierarchical object, or VectorizedDF
                compatible with self.get_tag("y_inner_mtype") format
            Case 1: self.get_tag("y_inner_mtype") supports scitype of y, then
                converted/coerced version of y, mtype determined by "y_inner_mtype" tag
            Case 2: self.get_tag("y_inner_mtype") does not support scitype of y, then
                VectorizedDF of y, iterated as the most complex supported scitype
                    (complexity order: Hierarchical > Panel > Series)
            Case 3: None if y was None
        X_inner : Series, Panel, or Hierarchical object, or VectorizedDF
                compatible with self.get_tag("X_inner_mtype") format
            Case 1: self.get_tag("X_inner_mtype") supports scitype of X, then
                converted/coerced version of X, mtype determined by "X_inner_mtype" tag
            Case 2: self.get_tag("X_inner_mtype") does not support scitype of X, then
                VectorizedDF of X, iterated as the most complex supported scitype
            Case 3: None if X was None

        Raises
        ------
        TypeError if y or X is not one of the permissible Series mtypes
        TypeError if y is not compatible with self.get_tag("scitype:y")
            if tag value is "univariate", y must be univariate
            if tag value is "multivariate", y must be bi- or higher-variate
            if tag value is "both", y can be either
        TypeError if self.get_tag("X-y-must-have-same-index") is True
            and the index set of X is not a super-set of the index set of y

        Writes to self
        --------------
        _y_mtype_last_seen : str, mtype of y
        _converter_store_y : dict, metadata from conversion for back-conversion
        """
        if X is None and y is None:
            return None, None

        def _most_complex_scitype(scitypes):
            """Return most complex scitype in a list of str."""
            if "Hierarchical" in scitypes:
                return "Hierarchical"
            elif "Panel" in scitypes:
                return "Panel"
            elif "Series" in scitypes:
                return "Series"
            else:
                raise ValueError("no series scitypes supported, bug in estimator")

        # retrieve supported mtypes
        y_inner_mtype = _coerce_to_list(self.get_tag("y_inner_mtype"))
        X_inner_mtype = _coerce_to_list(self.get_tag("X_inner_mtype"))
        y_inner_scitype = mtype_to_scitype(y_inner_mtype, return_unique=True)
        X_inner_scitype = mtype_to_scitype(X_inner_mtype, return_unique=True)

        ALLOWED_SCITYPES = ["Series", "Panel", "Hierarchical"]

        # checking y
        if y is not None:
            y_valid, _, y_metadata = check_is_scitype(
                y, scitype=ALLOWED_SCITYPES, return_metadata=True, var_name="y"
            )
            msg = (
                "y must be in an sktime compatible format, "
                "of scitype Series, Panel or Hierarchical, "
                "for instance a pandas.DataFrame with sktime compatible time indices, "
                "or with MultiIndex and lowest level a sktime compatible time index. "
                "See the forecasting tutorial examples/01_forecasting.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb"
            )
            if not y_valid:
                raise TypeError(msg)

            y_scitype = y_metadata["scitype"]
            self._y_mtype_last_seen = y_metadata["mtype"]

            requires_vectorization = y_scitype not in y_inner_scitype

            if (
                self.get_tag("scitype:y") == "univariate"
                and not y_metadata["is_univariate"]
            ):
                raise ValueError(
                    "y must be univariate, but found more than one variable"
                )
            if (
                self.get_tag("scitype:y") == "multivariate"
                and y_metadata["is_univariate"]
            ):
                raise ValueError(
                    "y must have two or more variables, but found only one"
                )
        else:
            # y_scitype is used below - set to None if y is None
            y_scitype = None
        # end checking y

        # checking X
        if X is not None:
            X_valid, _, X_metadata = check_is_scitype(
                X, scitype=ALLOWED_SCITYPES, return_metadata=True, var_name="X"
            )

            msg = (
                "X must be either None, or in an sktime compatible format, "
                "of scitype Series, Panel or Hierarchical, "
                "for instance a pandas.DataFrame with sktime compatible time indices, "
                "or with MultiIndex and lowest level a sktime compatible time index. "
                "See the forecasting tutorial examples/01_forecasting.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb"
            )
            if not X_valid:
                raise TypeError(msg)

            X_scitype = X_metadata["scitype"]
            requires_vectorization = X_scitype not in X_inner_scitype
        else:
            # X_scitype is used below - set to None if X is None
            X_scitype = None
        # end checking X

        # compatibility checks between X and y
        if X is not None and y is not None:
            if self.get_tag("X-y-must-have-same-index"):
                check_equal_time_index(X, y, mode="contains")

            if y_scitype != X_scitype:
                raise TypeError("X and y must have the same scitype")
        # end compatibility checking X and y

        # todo: add tests that :
        #   y_inner_scitype are same as X_inner_scitype
        #   y_inner_scitype always includes "less index" scitypes

        # convert X & y to supported inner type, if necessary
        #####################################################

        # convert X and y to a supported internal mtype
        #  it X/y mtype is already supported, no conversion takes place
        #  if X/y is None, then no conversion takes place (returns None)
        #  if vectorization is required, we wrap in Vect

        if not requires_vectorization:
            # converts y, skips conversion if already of right type
            y_inner = convert_to(
                y,
                to_type=y_inner_mtype,
                as_scitype=y_scitype,  # we are dealing with series
                store=self._converter_store_y,
                store_behaviour="reset",
            )

            # converts X, converts None to None if X is None
            X_inner = convert_to(
                X,
                to_type=X_inner_mtype,
                as_scitype=X_scitype,  # we are dealing with series
            )
        else:
            iterate_as = _most_complex_scitype(y_inner_scitype)
            if y is not None:
                y_inner = VectorizedDF(X=y, iterate_as=iterate_as, is_scitype=y_scitype)
            else:
                y_inner = None
            if X is not None:
                X_inner = VectorizedDF(X=X, iterate_as=iterate_as, is_scitype=X_scitype)
            else:
                X_inner = None

        return X_inner, y_inner

    def _check_X(self, X=None):
        """Shorthand for _check_X_y with one argument X, see _check_X_y."""
        return self._check_X_y(X=X)[0]

    def _update_X(self, X, enforce_index_type=None):
        if X is not None:
            X = check_X(X, enforce_index_type=enforce_index_type)
            if X is len(X) > 0:
                self._X = X.combine_first(self._X)

    def _update_y_X(self, y, X=None, enforce_index_type=None):
        """Update internal memory of seen training data.

        Accesses in self:
        _y : only if exists, then assumed same type as y and same cols
        _X : only if exists, then assumed same type as X and same cols
            these assumptions should be guaranteed by calls

        Writes to self:
        _y : same type as y - new rows from y are added to current _y
            if _y does not exist, stores y as _y
        _X : same type as X - new rows from X are added to current _X
            if _X does not exist, stores X as _X
            this is only done if X is not None
        cutoff : is set to latest index seen in y

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Endogenous time series
        X : pd.DataFrame or 2D np.ndarray, optional (default=None)
            Exogeneous time series
        """
        # we only need to modify _y if y is not None
        if y is not None:
            # if y is vectorized, unwrap it first
            if isinstance(y, VectorizedDF):
                y = y.X
            # we want to ensure that y is either numpy (1D, 2D, 3D)
            # or in one of the long pandas formats
            y = convert_to(
                y,
                to_type=[
                    "pd.DataFrame",
                    "pd.Series",
                    "np.ndarray",
                    "pd-multiindex",
                    "numpy3D",
                    "pd_multiindex_hier",
                ],
            )
            # if _y does not exist yet, initialize it with y
            if not hasattr(self, "_y") or self._y is None or not self.is_fitted:
                self._y = y
            # otherwise, update _y with the new rows in y
            #  if y is np.ndarray, we assume all rows are new
            elif isinstance(y, np.ndarray):
                # if 1D or 2D, axis 0 is "time"
                if y.ndim in [1, 2]:
                    self._y = np.concatenate(self._y, y, axis=0)
                # if 3D, axis 2 is "time"
                elif y.ndim == 3:
                    self._y = np.concatenate(self._y, y, axis=2)
            #  if y is pandas, we use combine_first to update
            elif isinstance(y, (pd.Series, pd.DataFrame)) and len(y) > 0:
                self._y = y.combine_first(self._y)

            # set cutoff to the end of the observation horizon
            self._set_cutoff_from_y(y)

        # we only need to modify _X if X is not None
        if X is not None:
            # if X is vectorized, unwrap it first
            if isinstance(X, VectorizedDF):
                X = X.X
            # we want to ensure that X is either numpy (1D, 2D, 3D)
            # or in one of the long pandas formats
            X = convert_to(
                X,
                to_type=[
                    "pd.DataFrame",
                    "np.ndarray",
                    "pd-multiindex",
                    "numpy3D",
                    "pd_multiindex_hier",
                ],
            )
            # if _X does not exist yet, initialize it with X
            if not hasattr(self, "_X") or self._X is None or not self.is_fitted:
                self._X = X
            # otherwise, update _X with the new rows in X
            #  if X is np.ndarray, we assume all rows are new
            elif isinstance(X, np.ndarray):
                # if 1D or 2D, axis 0 is "time"
                if X.ndim in [1, 2]:
                    self._X = np.concatenate(self._X, X, axis=0)
                # if 3D, axis 2 is "time"
                elif X.ndim == 3:
                    self._X = np.concatenate(self._X, X, axis=2)
            #  if X is pandas, we use combine_first to update
            elif isinstance(X, pd.DataFrame) and len(X) > 0:
                self._X = X.combine_first(self._X)

    def _get_y_pred(self, y_in_sample, y_out_sample):
        """Combine in- & out-sample prediction, slices given fh.

        Parameters
        ----------
        y_in_sample : pd.Series
            In-sample prediction
        y_out_sample : pd.Series
            Out-sample prediction

        Returns
        -------
        pd.Series
            y_pred, sliced by fh
        """
        y_pred = y_in_sample.append(y_out_sample, ignore_index=True).rename("y_pred")
        y_pred = pd.DataFrame(y_pred)
        # Workaround for slicing with negative index
        y_pred["idx"] = [x for x in range(-len(y_in_sample), len(y_out_sample))]
        y_pred = y_pred.loc[y_pred["idx"].isin(self.fh.to_indexer(self.cutoff).values)]
        y_pred.index = self.fh.to_absolute(self.cutoff)
        y_pred = y_pred["y_pred"].rename(None)
        return y_pred

    @property
    def cutoff(self):
        """Cut-off = "present time" state of forecaster.

        Returns
        -------
        cutoff : pandas compatible index element
        """
        return self._cutoff

    def _set_cutoff(self, cutoff):
        """Set and update cutoff.

        Parameters
        ----------
        cutoff: pandas compatible index element

        Notes
        -----
        Set self._cutoff is to `cutoff`.
        """
        self._cutoff = cutoff

    def _set_cutoff_from_y(self, y):
        """Set and update cutoff from series y.

        Parameters
        ----------
        y : sktime compatible time series data container
            must be of one of the following mtypes:
                pd.Series, pd.DataFrame, np.ndarray, of Series scitype
                pd.multiindex, numpy3D, nested_univ, df-list, of Panel scitype
                pd_multiindex_hier, of Hierarchical scitype
        Notes
        -----
        Set self._cutoff to latest index seen in `y`.
        """
        cutoff_idx = get_cutoff(y, self.cutoff)
        self._cutoff = cutoff_idx

    @contextmanager
    def _detached_cutoff(self):
        """Detached cutoff mode.

        When in detached cutoff mode, the cutoff can be updated but will
        be reset to the initial value after leaving the detached cutoff mode.

        This is useful during rolling-cutoff forecasts when the cutoff needs
        to be repeatedly reset, but afterwards should be restored to the
        original value.
        """
        cutoff = self.cutoff  # keep initial cutoff
        try:
            yield
        finally:
            # re-set cutoff to initial value
            self._set_cutoff(cutoff)

    @property
    def fh(self):
        """Forecasting horizon that was passed."""
        # raise error if some method tries to accessed it before it has been set
        if self._fh is None:
            raise ValueError(
                "No `fh` has been set yet, please specify `fh` " "in `fit` or `predict`"
            )

        return self._fh

    def _check_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Called from all methods where fh can be passed:
            fit, predict-like, update-like

        Reads and writes to self._fh
        Writes fh to self._fh if does not exist
        Checks equality of fh with self._fh if exists, raises error if not equal

        Parameters
        ----------
        fh : None, int, list, np.ndarray or ForecastingHorizon

        Returns
        -------
        self._fh : ForecastingHorizon or None
            if ForecastingHorizon, last passed fh coerced to ForecastingHorizon

        Raises
        ------
        ValueError if self._fh exists and is inconsistent with fh
        ValueError if fh is not passed (None) in a case where it must be:
            - in fit, if self has the tag "requires-fh-in-fit" (value True)
            - in predict, if it has not been passed in fit
        """
        requires_fh = self.get_tag("requires-fh-in-fit")

        msg = (
            f"This is because fitting of the `"
            f"{self.__class__.__name__}` "
            f"depends on `fh`. "
        )

        # below loop treats four cases from three conditions:
        #  A. forecaster is fitted yes/no - self.is_fitted
        #  B. no fh is passed yes/no - fh is None
        #  C. fh is optional in fit yes/no - optfh

        # B. no fh is passed
        if fh is None:
            # A. strategy fitted (call of predict or similar)
            if self._is_fitted:
                # in case C. fh is optional in fit:
                # if there is none from before, there is none overall - raise error
                if not requires_fh and self._fh is None:
                    raise ValueError(
                        "The forecasting horizon `fh` must be passed "
                        "either to `fit` or `predict`, "
                        "but was found in neither."
                    )
                # in case C. fh is not optional in fit: this is fine
                # any error would have already been caught in fit

            # A. strategy not fitted (call of fit)
            elif requires_fh:
                # in case fh is not optional in fit:
                # fh must be passed in fit
                raise ValueError(
                    "The forecasting horizon `fh` must be passed to "
                    "`fit`, but none was found. " + msg
                )
                # in case C. fh is optional in fit:
                # this is fine, nothing to check/raise

        # B. fh is passed
        else:
            # If fh is passed, validate (no matter the situation)
            fh = check_fh(fh)

            # fh is written to self if one of the following is true
            # - estimator has not been fitted yet (for safety from side effects)
            # - fh has not been seen yet
            # - fh has been seen, but was optional in fit,
            #     this means fh needs not be same and can be overwritten
            if not requires_fh or not self._fh or not self._is_fitted:
                self._fh = fh
            # there is one error condition:
            # - fh is mandatory in fit, i.e., fh in predict must be same if passed
            # - fh already passed, and estimator is fitted
            # - fh that was passed in fit is not the same as seen in predict
            # note that elif means: optfh == False, and self._is_fitted == True
            elif self._fh and not np.array_equal(fh, self._fh):
                # raise error if existing fh and new one don't match
                raise ValueError(
                    "A different forecasting horizon `fh` has been "
                    "provided from "
                    "the one seen in `fit`. If you want to change the "
                    "forecasting "
                    "horizon, please re-fit the forecaster. " + msg
                )
            # if existing one and new match, ignore new one

        return self._fh

    def _vectorize(self, methodname, **kwargs):
        """Vectorized/iterated loop over method of BaseForecaster.

        Uses forecasters_ attribute to store one forecaster per loop index.
        """
        PREDICT_METHODS = ["predict", "predict_quantiles"]

        if methodname == "fit":
            # create container for clones
            y = kwargs.pop("y")
            X = kwargs.pop("X", None)

            self._yvec = y

            idx = y.get_iter_indices()
            ys = y.as_list()

            if X is None:
                Xs = [None] * len(ys)
            else:
                Xs = X.as_list()

            self.forecasters_ = pd.DataFrame(index=idx, columns=["forecasters"])
            for i in range(len(idx)):
                self.forecasters_.iloc[i, 0] = clone(self)
                self.forecasters_.iloc[i, 0].fit(y=ys[i], X=Xs[i], **kwargs)

            return self
        elif methodname in PREDICT_METHODS:
            n = len(self.forecasters_.index)
            X = kwargs.pop("X", None)
            if X is None:
                Xs = [None] * n
            else:
                Xs = X.as_list()
            y_preds = []
            for i in range(n):
                method = getattr(self.forecasters_.iloc[i, 0], methodname)
                y_preds += [method(X=Xs[i], **kwargs)]
            y_pred = self._yvec.reconstruct(y_preds, overwrite_index=False)
            return y_pred

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

            core logic

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
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

            core logic

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.

        Returns
        -------
        y_pred : series of a type in self.get_tag("y_inner_mtype")
            Point forecasts at fh, with same index as fh
        """
        raise NotImplementedError("abstract method")

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        Writes to self:
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.

        Returns
        -------
        y_pred : series of a type in self.get_tag("y_inner_mtype")
            Point forecasts at fh, with same index as fh
        """
        if update_params:
            # default to re-fitting if update is not implemented
            warn(
                f"NotImplementedWarning: {self.__class__.__name__} "
                f"does not have a custom `update` method implemented. "
                f"{self.__class__.__name__} will be refit each time "
                f"`update` is called."
            )
            # we need to overwrite the mtype last seen, since the _y
            #    may have been converted
            mtype_last_seen = self._y_mtype_last_seen
            # refit with updated data, not only passed data
            self.fit(self._y, self._X, self.fh)
            # todo: should probably be self._fit, not self.fit
            # but looping to self.fit for now to avoid interface break
            self._y_mtype_last_seen = mtype_last_seen

        return self

    def _update_predict_single(
        self,
        y,
        fh,
        X=None,
        update_params=True,
    ):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.
        """
        self.update(y=y, X=X, update_params=update_params)
        return self.predict(fh=fh, X=X)

    def _predict_interval(self, fh, X=None, coverage=0.90):
        """Compute/return prediction interval forecasts.

        private _predict_interval containing the core logic,
            called from predict_interval and default _predict_quantiles

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        coverage : float or list, optional (default=0.95)
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
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_proba = self._has_implementation_of("_predict_proba")
        can_do_proba = implements_interval or implements_quantiles or implements_proba

        if not can_do_proba:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement "
                "probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                "This is likely a bug, please report, and/or set the flag to False."
            )

        if implements_quantiles:
            alphas = []
            for c in coverage:
                # compute quantiles corresponding to prediction interval coverage
                #  this uses symmetric predictive intervals
                alphas.extend([0.5 - 0.5 * float(c), 0.5 + 0.5 * float(c)])

            # compute quantile forecasts corresponding to upper/lower
            pred_int = self._predict_quantiles(fh=fh, X=X, alpha=alphas)

            # change the column labels (multiindex) to the format for intervals
            # idx returned by _predict_quantiles is
            #   2-level MultiIndex with variable names, alpha
            idx = pred_int.columns
            # variable names (unique, in same order)
            var_names = idx.get_level_values(0).unique()
            # if was univariate & unnamed variable, replace default
            if var_names == ["Quantiles"]:
                var_names = ["Coverage"]
            # idx returned by _predict_interval should be
            #   3-level MultiIndex with variable names, coverage, lower/upper
            int_idx = pd.MultiIndex.from_product(
                [var_names, coverage, ["lower", "upper"]]
            )

            pred_int.columns = int_idx

        return pred_int

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float, optional (default=[0.5])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_proba = self._has_implementation_of("_predict_proba")
        can_do_proba = implements_interval or implements_quantiles or implements_proba

        if not can_do_proba:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement "
                "probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                "This is likely a bug, please report, and/or set the flag to False."
            )

        if implements_interval:

            pred_int = pd.DataFrame()
            for a in alpha:
                # compute quantiles corresponding to prediction interval coverage
                #  this uses symmetric predictive intervals:
                coverage = abs(1 - 2 * a)

                # compute quantile forecasts corresponding to upper/lower
                pred_a = self._predict_interval(fh=fh, X=X, coverage=[coverage])
                pred_int = pd.concat([pred_int, pred_a], axis=1)

            # now we need to subset to lower/upper depending
            #   on whether alpha was < 0.5 or >= 0.5
            #   this formula gives the integer column indices giving lower/upper
            col_selector = (np.array(alpha) >= 0.5) + 2 * np.arange(len(alpha))
            pred_int = pred_int.iloc[:, col_selector]

            # change the column labels (multiindex) to the format for intervals
            # idx returned by _predict_interval is
            #   3-level MultiIndex with variable names, coverage, lower/upper
            idx = pred_int.columns
            # variable names (unique, in same order)
            var_names = idx.get_level_values(0).unique()
            # if was univariate & unnamed variable, replace default
            if var_names == ["Coverage"]:
                var_names = ["Quantiles"]
            # idx returned by _predict_quantiles should be
            #   is 2-level MultiIndex with variable names, alpha
            int_idx = pd.MultiIndex.from_product([var_names, alpha])

            pred_int.columns = int_idx

        return pred_int

    def _predict_var(
        self,
        fh=None,
        X=None,
    ):
        """Compute/return variance forecasts.

        private _predict_proba containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.

        Returns
        -------
        pred_var : pd.DataFrame
            Column names are exactly those of `y` passed in `fit`/`update`.
                For nameless formats, column index will be a RangeIndex.
            Row index is fh. Entries are variance forecasts, for var in col index.
        """
        from scipy.stats import norm

        # default behaviour is implemented if one of the following three is implemented
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_proba = self._has_implementation_of("_predict_proba")
        can_do_proba = implements_interval or implements_quantiles or implements_proba

        if not can_do_proba:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement "
                "probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                "This is likely a bug, please report, and/or set the flag to False."
            )

        if implements_proba:
            # todo: this works only univariate now, need to implement multivariate
            pred_var = self._predict_proba(fh=fh, X=X)
            pred_var = pd.DataFrame(pred_var)

            # ensure index and columns are as expected
            if fh.is_relative:
                fh = fh.to_absolute(self.cutoff)
            pred_var.index = fh.to_pandas()
            if isinstance(self._y, pd.DataFrame):
                pred_var.columns = self._y.columns

            return pred_var

        # if has one of interval/quantile predictions implemented:
        #   we get quantile forecasts for first and third quartile
        #   return variance of normal distribution with that first and third quartile
        if implements_interval or implements_quantiles:
            pred_int = self._predict_interval(fh=fh, X=X, coverage=[0.5])
            var_names = pred_int.columns.get_level_values(0).unique()
            vars_dict = {}
            for i in var_names:
                pred_int_i = pred_int[i].copy()
                # compute inter-quartile range (IQR), as pd.Series
                iqr_i = pred_int_i.iloc[:, 1] - pred_int_i.iloc[:, 0]
                # dividing by IQR of normal gives std of normal with same IQR
                std_i = iqr_i / (2 * norm.ppf(0.75))
                # and squaring gives variance (pd.Series)
                var_i = std_i ** 2
                vars_dict[i] = var_i

            # put together to pd.DataFrame
            #   the indices and column names are already correct
            pred_var = pd.DataFrame(vars_dict)

            # check whether column format was "nameless", set it to RangeIndex then
            if pred_var.columns == "Coverage":
                pred_var.columns = pd.RangeIndex(1)

        return pred_var

    # todo: does not work properly for multivariate or hierarchical
    #   still need to implement this - once interface is consolidated
    def _predict_proba(self, fh, X):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.

        Returns
        -------
        pred_dist : tfp Distribution object
            batch shape is 2D, of shape [len(fh), 1]
            event shape is 1D, with length equal to number of variables being forecast
            i-th (batch) distribution is forecast for i-th entry of fh
            j-th (event) component is j-th variable, same order as y in `fit`/`update`
        """
        import tensorflow_probability as tfp

        # default behaviour is implemented if one of the following three is implemented
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_var = self._has_implementation_of("_predict_var")
        can_do_proba = implements_interval or implements_quantiles or implements_var

        if not can_do_proba:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement "
                "probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                "This is likely a bug, please report, and/or set the flag to False."
            )

        # if any of the above are implemented, predict_var will have a default
        #   we use predict_var to get scale, and predict to get location
        pred_var = self._predict_var(fh=fh, X=X)
        pred_std = np.sqrt(pred_var)
        pred_mean = self.predict(fh=fh, X=X)
        # ensure that pred_mean is a pd.DataFrame
        df_types = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        pred_mean = convert_to(pred_mean, to_type=df_types)
        # pred_mean and pred_var now have the same format

        d = tfp.distributions.Normal
        pred_dist = d(loc=pred_mean, scale=pred_std)

        return pred_dist

    def _predict_moving_cutoff(
        self,
        y,
        cv,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Make single-step or multi-step moving cutoff predictions.

        Parameters
        ----------
        y : pd.Series
        cv : temporal cross-validation generator
        X : pd.DataFrame
        update_params : bool
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series
        """
        if return_pred_int:
            raise NotImplementedError()

        fh = cv.get_fh()
        y_preds = []
        cutoffs = []

        # enter into a detached cutoff mode
        with self._detached_cutoff():
            # set cutoff to time point before data
            self._set_cutoff(_shift(y.index[0], by=-1))
            # iterate over data
            for new_window, _ in cv.split(y):
                y_new = y.iloc[new_window]

                # we use `update_predict_single` here
                #  this updates the forecasting horizon
                y_pred = self._update_predict_single(
                    y_new,
                    fh,
                    X,
                    update_params=update_params,
                )
                if return_pred_int:
                    y_pred_int = self.predict_interval(fh, X, alpha=alpha)
                    y_pred_int = self._convert_new_to_old_pred_int(y_pred_int)
                    y_pred = (y_pred, y_pred_int)
                y_preds.append(y_pred)
                cutoffs.append(self.cutoff)
        return _format_moving_cutoff_predictions(y_preds, cutoffs)

    # TODO: remove in v0.11.0
    def _convert_new_to_old_pred_int(self, pred_int_new, alpha):
        name = pred_int_new.columns.get_level_values(0).unique()[0]
        alpha = check_alpha(alpha)
        alphas = [alpha] if isinstance(alpha, (float, int)) else alpha
        pred_int_old_format = [
            pd.DataFrame(
                {
                    "lower": pred_int_new[(name, a, "lower")],
                    "upper": pred_int_new[(name, a, "upper")],
                }
            )
            for a in alphas
        ]

        # for a single alpha, return single pd.DataFrame
        if len(alphas) == 1:
            return pred_int_old_format[0]

        # otherwise return list of pd.DataFrames
        return pred_int_old_format


def _format_moving_cutoff_predictions(y_preds, cutoffs):
    """Format moving-cutoff predictions.

    Parameters
    ----------
    y_preds: list of pd.Series or pd.DataFrames, of length n
            must have equal index and equal columns
    cutoffs: iterable of cutoffs, of length n

    Returns
    -------
    y_pred: pd.DataFrame, composed of entries of y_preds
        if length of elements in y_preds is 2 or larger:
            row-index = index common to the y_preds elements
            col-index = (cutoff[i], y_pred.column)
            entry is forecast at horizon given by row, from cutoff/variable at column
        if length of elements in y_preds is 1:
            row-index = forecasting horizon
            col-index = y_pred.column
    """
    # check that input format is correct
    if not isinstance(y_preds, list):
        raise ValueError(f"`y_preds` must be a list, but found: {type(y_preds)}")
    if len(y_preds) == 0:
        return pd.DataFrame(columns=cutoffs)
    if not isinstance(y_preds[0], (pd.DataFrame, pd.Series)):
        raise ValueError("y_preds must be a list of pd.Series or pd.DataFrame")
    ylen = len(y_preds[0])
    ytype = type(y_preds[0])
    if isinstance(y_preds[0], pd.DataFrame):
        ycols = y_preds[0].columns
    for y_pred in y_preds:
        if not isinstance(y_pred, ytype):
            raise ValueError("all elements of y_preds must be of the same type")
        if not len(y_pred) == ylen:
            raise ValueError("all elements of y_preds must be of the same length")
    if isinstance(y_preds[0], pd.DataFrame):
        for y_pred in y_preds:
            if not y_pred.columns.equals(ycols):
                raise ValueError("all elements of y_preds must have the same columns")

    if len(y_preds[0]) == 1:
        # return series for single step ahead predictions
        y_pred = pd.concat(y_preds)
    else:
        y_pred = pd.concat(y_preds, axis=1, keys=cutoffs)

    return y_pred
