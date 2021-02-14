#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning", "@big-o"]
__all__ = ["_SktimeForecaster", "_BaseWindowForecaster"]

from contextlib import contextmanager
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.model_selection import CutoffSplitter
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.utils.datetime import _shift
from sktime.utils.validation.forecasting import check_X
from sktime.utils.validation.forecasting import check_alpha
from sktime.utils.validation.forecasting import check_cv
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_y
from sktime.utils.validation.forecasting import check_y_X


class _SktimeForecaster(BaseForecaster):
    """Base class for forecaster implemented in sktime"""

    def __init__(self):
        # training data
        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh
        super(_SktimeForecaster, self).__init__()

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
        """Combining in-sample and out-sample prediction
        and slicing on given fh.

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
        """Combining lower and upper bound of
        prediction intervals. Slicing on fh.

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
        """The time point at which to make forecasts

        Returns
        -------
        cutoff : int
        """
        return self._cutoff

    def _set_cutoff(self, cutoff):
        """Set and update cutoff

        Parameters
        ----------
        cutoff : int
        """
        self._cutoff = cutoff

    @contextmanager
    def _detached_cutoff(self):
        """When in detached cutoff mode, the cutoff can be updated but will
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
        """The forecasting horizon"""
        # raise error if some method tries to accessed it before it has been
        # set
        if self._fh is None:
            raise ValueError(
                "No `fh` has been set yet, please specify `fh` " "in `fit` or `predict`"
            )
        return self._fh

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Abstract base method, implemented by mixin classes.

        Parameters
        ----------
        fh : None, int, list, np.array
        """
        raise NotImplementedError()

    def fit(self, y, X=None, fh=None):
        raise NotImplementedError("abstract method")

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Make forecasts

        Parameters
        ----------
        fh : int, list or np.array
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
        y_pred_int : pd.DataFrame
            Prediction intervals
        """
        self.check_is_fitted()
        self._set_fh(fh)
        return self._predict(self.fh, X, return_pred_int=return_pred_int, alpha=alpha)

    def compute_pred_int(self, y_pred, alpha=DEFAULT_ALPHA):
        """
        Get the prediction intervals for a forecast. Must be run *after* the
        forecaster has been fitted.

        If alpha is iterable, multiple intervals will be calculated.

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

        alphas = check_alpha(alpha)
        errors = self._compute_pred_err(alphas)

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

    def _compute_pred_err(self, alphas):
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
        raise NotImplementedError("abstract method")

    def update_predict_single(
        self,
        y_new,
        fh=None,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Update and make forecasts."

        This method is useful for updating forecasts in a single step,
        allowing to make use of more efficient
        updating algorithms than calling update and predict sequentially.

        Parameters
        ----------
        y_new : pd.Series
        fh : int, list or np.array
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

    def _update_predict_single(
        self,
        y,
        fh,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Internal method for updating and making forecasts.

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.
        """
        self.update(y, X, update_params=update_params)
        return self.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)

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
        X : pd.DataFrame
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        self.check_is_fitted()
        self._update_y_X(y, X)
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

        if return_pred_int:
            raise NotImplementedError()
        y = check_y(y)
        cv = check_cv(cv) if cv is not None else SlidingWindowSplitter(fh=self.fh)
        return self._predict_moving_cutoff(
            y,
            cv,
            X,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
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
        """Make single-step or multi-step moving cutoff predictions

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

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Internal predict

        Parameters
        ----------
        fh : np.array
        X : pd.DataFrame, optional (default=None)
        return_pred_int : bool
        alpha : float or list of floats

        Returns
        -------
        y_pred : pd.Series
        """
        raise NotImplementedError("abstract method")


class _OptionalForecastingHorizonMixin:
    """Mixin class for forecasters which can take the forecasting horizon
    either
    during fit or predict."""

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Parameters
        ----------
        fh : None, int, list or np.ndarray
        """
        if fh is None:
            if self.is_fitted:
                # if no fh passed and there is none already, raise error
                if self._fh is None:
                    raise ValueError(
                        "The forecasting horizon `fh` must be passed "
                        "either to `fit` or `predict`, "
                        "but was found in neither."
                    )
                # otherwise if no fh passed, but there is one already,
                # we can simply use that one
        else:
            # If fh is passed, validate first, then check if there is one
            # already,
            # and overwrite

            # A warning should only be raised if fh passed to fit is
            # overwritten, but no warning is required when no fh has been provided in
            # fit, and different fhs are passed to predict, but this requires
            # to keep track of whether fh has been passed to fit or not, hence not
            # implemented for cutoff.
            fh = check_fh(fh)
            self._fh = fh


class _RequiredForecastingHorizonMixin:
    """Mixin class for forecasters which require the forecasting horizon
    during fit."""

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Parameters
        ----------
        fh : None, int, list, np.ndarray
        """

        msg = (
            f"This is because fitting of the `"
            f"{self.__class__.__name__}` "
            f"depends on `fh`. "
        )

        if fh is None:
            if self.is_fitted:
                # intended workflow, no fh is passed when the forecaster is
                # already fitted
                pass
            else:
                # fh must be passed when forecaster is not fitted yet
                raise ValueError(
                    "The forecasting horizon `fh` must be passed to "
                    "`fit`, "
                    "but none was found. " + msg
                )
        else:
            fh = check_fh(fh)
            if self.is_fitted:
                if not np.array_equal(fh, self._fh):
                    # raise error if existing fh and new one don't match
                    raise ValueError(
                        "A different forecasting horizon `fh` has been "
                        "provided from "
                        "the one seen in `fit`. If you want to change the "
                        "forecasting "
                        "horizon, please re-fit the forecaster. " + msg
                    )
                # if existing one and new match, ignore new one
                pass
            else:
                # intended workflow: fh is passed when forecaster is not
                # fitted yet
                self._fh = fh


class _BaseWindowForecaster(_SktimeForecaster):
    """Base class for forecasters that use """

    def __init__(self, window_length=None):
        super(_BaseWindowForecaster, self).__init__()
        self.window_length = window_length
        self.window_length_ = None

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
        y_pred : pd.Series or pd.DataFrame
        """
        if cv is not None:
            cv = check_cv(cv)
        else:
            cv = SlidingWindowSplitter(
                self.fh.to_relative(self.cutoff), window_length=self.window_length_
            )
        return self._predict_moving_cutoff(
            y,
            cv,
            X,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Internal predict"""
        if return_pred_int:
            raise NotImplementedError()

        kwargs = {"X": X, "return_pred_int": return_pred_int, "alpha": alpha}

        # all values are out-of-sample
        if fh.is_all_out_of_sample(self.cutoff):
            return self._predict_fixed_cutoff(
                fh.to_out_of_sample(self.cutoff), **kwargs
            )

        # all values are in-sample
        elif fh.is_all_in_sample(self.cutoff):
            return self._predict_in_sample(fh.to_in_sample(self.cutoff), **kwargs)

        # both in-sample and out-of-sample values
        else:
            y_ins = self._predict_in_sample(fh.to_in_sample(self.cutoff), **kwargs)
            y_oos = self._predict_fixed_cutoff(
                fh.to_out_of_sample(self.cutoff), **kwargs
            )
            return y_ins.append(y_oos)

    def _predict_fixed_cutoff(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Make single-step or multi-step fixed cutoff predictions

        Parameters
        ----------
        fh : np.array
            all positive (> 0)
        X : pd.DataFrame
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series
        """
        # assert all(fh > 0)
        y_pred = self._predict_last_window(
            fh, X, return_pred_int=return_pred_int, alpha=alpha
        )
        index = fh.to_absolute(self.cutoff)
        return pd.Series(y_pred, index=index)

    def _predict_in_sample(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Make in-sample prediction using single-step moving-cutoff
        predictions

        Parameters
        ----------
        fh : np.array
            all non-positive (<= 0)
        X : pd.DataFrame
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred : pd.DataFrame or pd.Series
        """
        y_train = self._y

        # generate cutoffs from forecasting horizon, note that cutoffs are
        # still based on integer indexes, so that they can be used with .iloc
        cutoffs = fh.to_relative(self.cutoff) + len(y_train) - 2
        cv = CutoffSplitter(cutoffs, fh=1, window_length=self.window_length_)
        return self._predict_moving_cutoff(
            y_train,
            cv,
            X,
            update_params=False,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Internal predict

        Parameters
        ----------
        fh : np.array
        X : pd.DataFrame
        return_pred_int : bool
        alpha : float or list of floats

        Returns
        -------
        y_pred : np.array
        """
        raise NotImplementedError("abstract method")

    def _get_last_window(self):
        """Select last window"""
        # get the start and end points of the last window
        cutoff = self.cutoff
        start = _shift(cutoff, by=-self.window_length_ + 1)

        # get the last window of the endogenous variable
        y = self._y.loc[start:cutoff].to_numpy()

        # if exogenous variables are given, also get the last window of
        # those
        if self._X is not None:
            X = self._X.loc[start:cutoff].to_numpy()
        else:
            X = None
        return y, X

    @staticmethod
    def _predict_nan(fh):
        """Predict nan if predictions are not possible"""
        return np.full(len(fh), np.nan)

    def _update_predict_single(
        self,
        y,
        fh,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Internal method for updating and making forecasts.

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.

        Parameters
        ----------
        y
        fh
        X
        update_params
        return_pred_int
        alpha

        Returns
        -------

        """
        if X is not None:
            raise NotImplementedError()
        self.update(y, X, update_params=update_params)
        return self._predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)


def _format_moving_cutoff_predictions(y_preds, cutoffs):
    """Format moving-cutoff predictions"""
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
