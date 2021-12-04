# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Base class template for forecaster scitype.

    class name: BaseForecaster

Scitype defining methods:
    fitting            - fit(y, X=None, fh=None)
    forecasting        - predict(fh=None, X=None)
    fit&forecast       - fit_predict(y, X=None, fh=None)
    forecast intervals - predict_interval(fh=None, X=None, coverage=0.90)
    forecast quantiles - predict_quantiles(fh=None, X=None, alpha=[0.05, 0.95])
    updating           - update(y, X=None, update_params=True)
    update&forecast    - update_predict(cv=None, X=None, update_params=True)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

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

from sktime.base import BaseEstimator
from sktime.datatypes import convert_to, mtype
from sktime.utils.datetime import _shift
from sktime.utils.validation.forecasting import check_alpha, check_cv, check_fh, check_X
from sktime.utils.validation.series import check_equal_time_index, check_series

DEFAULT_ALPHA = 0.05


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
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        self._set_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X/y to the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        #####################################################

        self._fit(y=y_inner, X=X_inner, fh=fh)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(
        self,
        fh=None,
        X=None,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
        keep_old_return_type=True,
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
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y passed in fit (most recently)
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            in this case, return is 2-tuple (otherwise a single y_pred)
            Prediction intervals
        """
        # handle inputs

        self.check_is_fitted()
        self._set_fh(fh)

        # todo deprecate NotImplementedError in v 10.0.1
        if return_pred_int and not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "prediction intervals. Please set return_pred_int=False. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )

        # input check and conversion for X
        X_inner = self._check_X(X=X)

        # this is how it is supposed to be after the refactor is complete and effective
        if not return_pred_int:
            y_pred = self._predict(
                self.fh,
                X=X_inner,
            )

            # convert to output mtype, identical with last y mtype seen
            y_out = convert_to(
                y_pred,
                self._y_mtype_last_seen,
                as_scitype="Series",
                store=self._converter_store_y,
            )

            return y_out

        # keep following code for downward compatibility,
        # todo: can be deleted once refactor is completed and effective,
        # todo: deprecate in v 10
        else:
            warn(
                "return_pred_int in predict() will be deprecated;"
                "please use predict_interval() instead to generate "
                "prediction intervals.",
                FutureWarning,
            )

            if not self._has_predict_quantiles_been_refactored():
                # this means the method is not refactored
                y_pred = self._predict(
                    self.fh,
                    X=X_inner,
                    return_pred_int=return_pred_int,
                    alpha=alpha,
                )

                # returns old return type anyways
                pred_int = y_pred[1]
                y_pred = y_pred[0]

            else:
                # it's already refactored
                # opposite definition previously vs. now
                coverage = [1 - a for a in alpha]
                pred_int = self.predict_interval(fh=fh, X=X_inner, coverage=coverage)

                if keep_old_return_type:
                    pred_int = _convert_new_to_old_pred_int(pred_int, alpha)

            # convert to output mtype, identical with last y mtype seen
            y_out = convert_to(
                y_pred,
                self._y_mtype_last_seen,
                as_scitype="Series",
                store=self._converter_store_y,
            )

            return (y_out, pred_int)

    def fit_predict(
        self, y, X=None, fh=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
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
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            in this case, return is 2-tuple (otherwise a single y_pred)
            Prediction intervals
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        self._set_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X/y to the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # apply fit and then predict
        self._fit(y=y_inner, X=X_inner, fh=fh)
        self._is_fitted = True
        # call the public predict to avoid duplicating output conversions
        #  input conversions are skipped since we are using X_inner
        return self.predict(
            fh=fh, X=X_inner, return_pred_int=return_pred_int, alpha=alpha
        )

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
        if alpha is None:
            alpha = [0.05, 0.95]

        self.check_is_fitted()
        self._set_fh(fh)
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

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being quantile fractions for interval low-high.
                Quantile fractions are 0.5 - c/2, 0.5 + c/2 for c in coverage.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        # input check for X

        self._set_fh(fh)
        X_inner = self._check_X(X=X)
        self.check_is_fitted()

        coverage = check_alpha(coverage)
        pred_int = self._predict_interval(fh=fh, X=X_inner, coverage=coverage)
        return pred_int

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
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
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
        return_pred_int : bool, optional (default=False)
        alpha : int or list of ints, optional (default=None)

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            in this case, return is 2-tuple (otherwise a single y_pred)
            Prediction intervals
        """
        self.check_is_fitted()

        if return_pred_int and not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "prediction intervals. Please set return_pred_int=False. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        cv = check_cv(cv)

        return self._predict_moving_cutoff(
            y=y_inner,
            cv=cv,
            X=X_inner,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    def update_predict_single(
        self,
        y=None,
        y_new=None,
        fh=None,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
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
        y_new : alias for y for downwards compatibility, pass only one of y, y_new
            to be deprecated in version 0.10.0
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to and to predict from
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain y.index and fh.index
        update_params : bool, optional (default=False)
        return_pred_int : bool, optional (default=False)
            If True, prediction intervals are returned in addition to point
            predictions.
        alpha : float or list of floats

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y
        pred_ints : pd.DataFrame
            Prediction intervals
        """
        # todo deprecate return_pred_int in v 0.10.1
        self.check_is_fitted()
        self._set_fh(fh)

        # handle input alias, deprecate in v 0.10.1
        if y is None:
            y = y_new
        if y is None:
            raise ValueError("y must be of Series type and cannot be None")

        self.check_is_fitted()
        self._set_fh(fh)

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update internal _X/_y with the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        return self._update_predict_single(
            y=y_inner,
            fh=self.fh,
            X=X_inner,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

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
        y_inner : Series compatible with self.get_tag("y_inner_mtype") format
            converted/coerced version of y, mtype determined by "y_inner_mtype" tag
            None if y was None
        X_inner : Series compatible with self.get_tag("X_inner_mtype") format
            converted/coerced version of y, mtype determined by "X_inner_mtype" tag
            None if X was None

        Raises
        ------
        TypeError if y or X is not one of the permissible Series mtypes
        TypeError if y is not compatible with self.get_tag("scitype:y")
            if tag value is "univariate", y must be univariate
            if tag value is "multivariate", y must be bi- or higher-variate
            if tag vaule is "both", y can be either
        TypeError if self.get_tag("X-y-must-have-same-index") is True
            and the index set of X is not a super-set of the index set of y

        Writes to self
        --------------
        _y_mtype_last_seen : str, mtype of y
        _converter_store_y : dict, metadata from conversion for back-conversion
        """
        # input checks and minor coercions on X, y
        ###########################################

        enforce_univariate = self.get_tag("scitype:y") == "univariate"
        enforce_multivariate = self.get_tag("scitype:y") == "multivariate"
        enforce_index_type = self.get_tag("enforce_index_type")

        # checking y
        if y is not None:
            check_y_args = {
                "enforce_univariate": enforce_univariate,
                "enforce_multivariate": enforce_multivariate,
                "enforce_index_type": enforce_index_type,
                "allow_None": False,
                "allow_empty": True,
            }

            y = check_series(y, **check_y_args, var_name="y")

            self._y_mtype_last_seen = mtype(y)
        # end checking y

        # checking X
        if X is not None:
            X = check_series(X, enforce_index_type=enforce_index_type, var_name="X")
            if self.get_tag("X-y-must-have-same-index"):
                check_equal_time_index(X, y)
        # end checking X

        # convert X & y to supported inner type, if necessary
        #####################################################

        # retrieve supported mtypes

        # convert X and y to a supported internal mtype
        #  it X/y mtype is already supported, no conversion takes place
        #  if X/y is None, then no conversion takes place (returns None)
        y_inner_mtype = self.get_tag("y_inner_mtype")
        y_inner = convert_to(
            y,
            to_type=y_inner_mtype,
            as_scitype="Series",  # we are dealing with series
            store=self._converter_store_y,
        )

        X_inner_mtype = self.get_tag("X_inner_mtype")
        X_inner = convert_to(
            X,
            to_type=X_inner_mtype,
            as_scitype="Series",  # we are dealing with series
        )

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
        y : pd.Series, pd.DataFrame, or nd.nparray (1D or 2D)
            Endogenous time series
        X : pd.DataFrame or 2D np.ndarray, optional (default=None)
            Exogeneous time series
        """
        # we only need to modify _y if y is not None
        if y is not None:
            # if _y does not exist yet, initialize it with y
            if not hasattr(self, "_y") or self._y is None or not self.is_fitted:
                self._y = y
            # otherwise, update _y with the new rows in y
            #  if y is np.ndarray, we assume all rows are new
            elif isinstance(y, np.ndarray):
                self._y = np.concatenate(self._y, y)
            #  if y is pandas, we use combine_first to update
            elif isinstance(y, (pd.Series, pd.DataFrame)) and len(y) > 0:
                self._y = y.combine_first(self._y)

            # set cutoff to the end of the observation horizon
            self._set_cutoff_from_y(y)

        # we only need to modify _X if X is not None
        if X is not None:
            # if _X does not exist yet, initialize it with X
            if not hasattr(self, "_X") or self._X is None or not self.is_fitted:
                self._X = X
            # otherwise, update _X with the new rows in X
            #  if X is np.ndarray, we assume all rows are new
            elif isinstance(X, np.ndarray):
                self._X = np.concatenate(self._X, X)
            #  if X is pandas, we use combine_first to update
            elif isinstance(X, (pd.Series, pd.DataFrame)) and len(X) > 0:
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
        cutoff : int
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
        y: pd.Series, pd.DataFrame, or np.array
            Time series from which to infer the cutoff.

        Notes
        -----
        Set self._cutoff to last index seen in `y`.
        """
        y_mtype = mtype(y, as_scitype="Series")

        if len(y) > 0:
            if y_mtype in ["pd.Series", "pd.DataFrame"]:
                self._cutoff = y.index[-1]
            elif y_mtype == "np.ndarray":
                self._cutoff = len(y)
            else:
                raise TypeError("y does not have a supported type")

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

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Parameters
        ----------
        fh : None, int, list, np.ndarray or ForecastingHorizon
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
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
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
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
            - Will be removed in v 0.10.0
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : series of a type in self.get_tag("y_inner_mtype")
            Point forecasts at fh, with same index as fh
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals - deprecate in v 0.10.1

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
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : series of a type in self.get_tag("y_inner_mtype")
            Point forecasts at fh, with same index as fh
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        if update_params:
            # default to re-fitting if update is not implemented
            warn(
                f"NotImplementedWarning: {self.__class__.__name__} "
                f"does not have a custom `update` method implemented. "
                f"{self.__class__.__name__} will be refit each time "
                f"`update` is called."
            )
            # refit with updated data, not only passed data
            self.fit(self._y, self._X, self.fh)
            # todo: should probably be self._fit, not self.fit
            # but looping to self.fit for now to avoid interface break

        return self

    def _update_predict_single(
        self,
        y,
        fh,
        X=None,
        update_params=True,
        # todo: deprecate return_pred_int in v 10.0.1
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.
        """
        self.update(y, X, update_params=update_params)
        return self.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction interval forecasts.

        If coverage is iterable, multiple intervals will be calculated.

            core logic

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
           Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
           Exogenous time series
        alpha : float or list, optional (default=0.95)
           Probability mass covered by interval or list of coverages.

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being quantile fractions for interval low-high.
                Quantile fractions are 0.5 - c/2, 0.5 + c/2 for c in coverage.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        alphas = []
        for c in coverage:
            alphas.extend([(1 - c) / 2, 0.5 + (c / 2)])
        alphas.sort()
        pred_int = self._predict_quantiles(fh=fh, X=X, alpha=alphas)
        return pred_int

    def _predict_quantiles(self, fh, X, alpha):
        """
        Compute/return prediction quantiles for a forecast.

        Must be run *after* the forecaster has been fitted.

        If alpha is iterable, multiple quantiles will be calculated.

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have the capability to return "
            "prediction quantiles. If you "
            "think this estimator should have the capability, please open "
            "an issue on sktime."
        )

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
                    return_pred_int=return_pred_int,
                    alpha=alpha,
                )
                y_preds.append(y_pred)
                cutoffs.append(self.cutoff)
        return _format_moving_cutoff_predictions(y_preds, cutoffs)

    # TODO: remove in v0.10.0
    def _has_predict_quantiles_been_refactored(self):
        if "_predict_quantiles" in type(self).__dict__.keys():
            return True
        else:
            return False


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


# TODO: remove in v0.10.0
def _convert_new_to_old_pred_int(pred_int_new, alpha):
    name = pred_int_new.columns.get_level_values(0).unique()[0]
    alpha = check_alpha(alpha)
    pred_int_old_format = [
        pd.DataFrame(
            {
                "lower": pred_int_new[name, a / 2],
                "upper": pred_int_new[name, 1 - (a / 2)],
            }
        )
        for a in alpha
    ]

    # for a single alpha, return single pd.DataFrame
    if len(alpha) == 1:
        return pred_int_old_format[0]

    # otherwise return list of pd.DataFrames
    return pred_int_old_format
