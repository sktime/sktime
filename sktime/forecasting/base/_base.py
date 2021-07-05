# -*- coding: utf-8 -*-
"""
Base class template for forecaster scitype.

    class name: BaseForecaster

Scitype defining methods:
    fitting         - fit(self, y, X=None, fh=None)
    forecasting     - predict(self, fh=None, X=None, return_pred_int=False,
                              alpha=DEFAULT_ALPHA)
    fit&forecast    - fit_predict(self, y, X=None, fh=None,
                              return_pred_int=False, alpha=DEFAULT_ALPHA)
    updating        - update(self, y, X=None, update_params=True)
    update&forecast - update_predict(y, cv=None, X=None, update_params=True,
                        return_pred_int=False, alpha=DEFAULT_ALPHA)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""


__author__ = ["Markus Löning", "@big-o", "fkiraly"]
__all__ = ["BaseForecaster"]

from sktime.base import BaseEstimator

from contextlib import contextmanager
from warnings import warn

import numpy as np
import pandas as pd

from sktime.utils.datetime import _shift
from sktime.utils.validation.forecasting import check_X
from sktime.utils.validation.forecasting import check_alpha
from sktime.utils.validation.forecasting import check_cv
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_y
from sktime.utils.validation.forecasting import check_y_X


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
        "requires-fh-in-fit": True,  # is forecasting horizon already required in fit?
        "handles-missing-data": False,  # can estimator handle missing data?
        "univariate-only": True,  # can estimator deal with multivariate series y?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
    }

    def __init__(self):
        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        super(BaseForecaster, self).__init__()

    def fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogeneous data
        Returns
        -------
        self : reference to self.

        State change
        ------------
        stores data in self._X and self._y
        stores fh, if passed
        updates self.cutoff to most recent time in y
        creates fitted model (attributes ending in "_")
        sets is_fitted flag to true
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        self._set_fh(fh)
        y, X = check_y_X(y, X)

        self._X = X
        self._y = y

        self._set_cutoff(y.index[-1])

        self._fit(y=y, X=X, fh=fh)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        self.check_is_fitted()
        self._set_fh(fh)

        # todo: check_X should let a None argument pass here, but it doesn't
        if X is not None:
            X = check_X(X)

        # this should be here, but it breaks the ARIMA forecasters
        #  that is because check_alpha converts to list, but ARIMA forecaster
        #  doesn't do the check, and needs it as a float or it breaks
        # todo: needs fixing in ARIMA and AutoARIMA
        # alpha = check_alpha(alpha)

        return self._predict(self.fh, X, return_pred_int=return_pred_int, alpha=alpha)

    def fit_predict(
        self, y, X=None, fh=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Fit and forecast time series at future horizon.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """

        self.fit(y=y, X=X, fh=fh)

        return self._predict(fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha)

    def compute_pred_int(self, y_pred, alpha=DEFAULT_ALPHA):
        """
        Compute/return prediction intervals for a forecast.

        Must be run *after* the forecaster has been fitted.

        If alpha is iterable, multiple intervals will be calculated.

        public method including checks & utility
        dispatches to core logic in _compute_pred_int

        Parameters
        ----------
        y_pred : pd.Series
            Point predictions.
        alpha : float or list, optional (default=0.95)
            A significance level or list of significance levels.

        Returns
        -------
        intervals : pd.DataFrame
            A table of upper and lower bounds for each point prediction in
            ``y_pred``. If ``alpha`` was iterable, then ``intervals`` will be a
            list of such tables.
        """
        self.check_is_fitted()
        alphas = check_alpha(alpha)
        errors = self._compute_pred_int(alphas)

        # compute prediction intervals
        pred_int = [
            pd.DataFrame({"lower": y_pred - error, "upper": y_pred + error})
            for error in errors
        ]

        # for a single alpha, return single pd.DataFrame
        if isinstance(alpha, float):
            return pred_int[0]

        # otherwise return list of pd.DataFrames
        return pred_int

    def update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        This is useful in an online learning setting where new data is observed as
        time moves on. Updating the cutoff value allows to generate new predictions
        from the most recent time point that was observed. Updating the fitted
        parameters allows to incrementally update the parameters without having to
        completely refit. However, note that if no estimator-specific update method
        has been implemented for updating parameters refitting is the default fall-back
        option.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogeneous data
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self

        State change
        ------------
        updates self._X and self._y with new data
        updates self.cutoff to most recent time in y
        if update_params=True, updates model (attributes ending in "_")
        """
        self.check_is_fitted()
        self._update_y_X(y, X)

        self._update(y=y, X=X, update_params=update_params)

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
        """Make and update predictions iteratively over the test set.

        Parameters
        ----------
        y : pd.Series
        cv : temporal cross-validation generator, optional (default=None)
        X : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=True)
        return_pred_int : bool, optional (default=False)
        alpha : int or list of ints, optional (default=None)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame
            Prediction intervals
        """
        self.check_is_fitted()

        if return_pred_int:
            raise NotImplementedError()
        y = check_y(y)
        cv = check_cv(cv)

        return self._predict_moving_cutoff(
            y,
            cv,
            X,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    def update_predict_single(
        self,
        y_new,
        fh=None,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Update and make forecasts.

        This method is useful for updating forecasts in a single step,
        allowing to make use of more efficient
        updating algorithms than calling update and predict sequentially.

        Parameters
        ----------
        y_new : pd.Series
        fh : int, list, np.array or ForecastingHorizon
        X : pd.DataFrame
        update_params : bool, optional (default=False)
        return_pred_int : bool, optional (default=False)
            If True, prediction intervals are returned in addition to point
            predictions.
        alpha : float or list of floats

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        pred_ints : pd.DataFrame
            Prediction intervals
        """
        self.check_is_fitted()
        self._set_fh(fh)
        return self._update_predict_single(
            y_new,
            self.fh,
            X,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    def score(self, y, X=None, fh=None):
        """Scores forecast against ground truth, using MAPE.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to compare the forecasts.
        fh : int, list, array-like or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.

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

        Returns
        -------
        fitted_params : dict
        """
        raise NotImplementedError("abstract method")

    def _set_y_X(self, y, X=None, enforce_index_type=None):
        """Set training data.

        Parameters
        ----------
        y : pd.Series
            Endogenous time series
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        """
        # set initial training data
        self._y, self._X = check_y_X(
            y, X, allow_empty=False, enforce_index_type=enforce_index_type
        )

        # set initial cutoff to the end of the training data
        self._set_cutoff(y.index[-1])

    def _update_X(self, X, enforce_index_type=None):
        if X is not None:
            X = check_X(X, enforce_index_type=enforce_index_type)
            if X is len(X) > 0:
                self._X = X.combine_first(self._X)

    def _update_y_X(self, y, X=None, enforce_index_type=None):
        """Update training data.

        Parameters
        ----------
        y : pd.Series
            Endogenous time series
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        """
        # update only for non-empty data
        y, X = check_y_X(y, X, allow_empty=True, enforce_index_type=enforce_index_type)

        if len(y) > 0:
            self._y = y.combine_first(self._y)

            # set cutoff to the end of the observation horizon
            self._set_cutoff(y.index[-1])

            # update X if given
            if X is not None:
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

    def _get_pred_int(self, lower, upper):
        """Combine lower/upper bounds of pred.intervals, slice on fh.

        Parameters
        ----------
        lower : pd.Series
            Lower bound (can contain also in-sample bound)
        upper : pd.Series
            Upper bound (can contain also in-sample bound)

        Returns
        -------
        pd.DataFrame
            pred_int, predicion intervalls (out-sample, sliced by fh)
        """
        pred_int = pd.DataFrame({"lower": lower, "upper": upper})
        # Out-sample fh
        fh_out = self.fh.to_out_of_sample(cutoff=self.cutoff)
        # If pred_int contains in-sample prediction intervals
        if len(pred_int) > len(self._y):
            len_out = len(pred_int) - len(self._y)
            # Workaround for slicing with negative index
            pred_int["idx"] = [x for x in range(-len(self._y), len_out)]
        # If pred_int does not contain in-sample prediction intervals
        else:
            pred_int["idx"] = [x for x in range(len(pred_int))]
        pred_int = pred_int.loc[
            pred_int["idx"].isin(fh_out.to_indexer(self.cutoff).values)
        ]
        pred_int.index = fh_out.to_absolute(self.cutoff)
        pred_int = pred_int.drop(columns=["idx"])
        return pred_int

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
        cutoff : int
        """
        self._cutoff = cutoff

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
        requires_fh = self._all_tags().get("requires-fh-in-fit", True)

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

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

            core logic

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        raise NotImplementedError("abstract method")

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

            core logic

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals

        State change
        ------------
        updates self._X and self._y with new data
        updates self.cutoff to most recent time in y
        if update_params=True, updates model (attributes ending in "_")
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

    def _compute_pred_int(self, alphas):
        """Calculate the prediction errors for each point.

        Parameters
        ----------
        alpha : float or list, optional (default=0.95)
            A significance level or list of significance levels.

        Returns
        -------
        errors : list of pd.Series
            Each series in the list will contain the errors for each point in
            the forecast for the corresponding alpha.
        """

        # this should be the NotImplementedError
        # but current interface assumes private method
        # _compute_pred_err(alphas), not _compute_pred_int
        # so looping this through in order for existing classes to work
        return self._compute_pred_err(alphas)

        # todo: fix this in descendants, and change to
        # raise NotImplementedError("abstract method")

    def _compute_pred_err(self, alphas):
        """ temporary loopthrough for _compute_pred_err"""
        raise NotImplementedError("abstract method")

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

                # we cannot use `update_predict_single` here, as this would
                # re-set the forecasting horizon, instead we use
                # the internal `_update_predict_single` method
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


def _format_moving_cutoff_predictions(y_preds, cutoffs):
    """Format moving-cutoff predictions."""
    if not isinstance(y_preds, list):
        raise ValueError(f"`y_preds` must be a list, but found: {type(y_preds)}")

    if len(y_preds[0]) == 1:
        # return series for single step ahead predictions
        return pd.concat(y_preds)

    else:
        # return data frame when we predict multiple steps ahead
        y_pred = pd.DataFrame(y_preds).T
        y_pred.columns = cutoffs
        if y_pred.shape[1] == 1:
            return y_pred.iloc[:, 0]
        return y_pred
