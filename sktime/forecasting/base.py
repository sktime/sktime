#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning", "@big-o"]
__all__ = ["BaseForecaster", "RequiredForecastingHorizonMixin",
           "OptionalForecastingHorizonMixin", "DEFAULT_ALPHA"]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.exceptions import NotFittedError
from sktime.utils.plotting import composite_alpha
from sktime.utils.validation.forecasting import check_cv
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_y
from contextlib import contextmanager
from sktime.forecasting.model_selection import ManualWindowSplitter

DEFAULT_ALPHA = 0.05


class BaseForecaster(BaseEstimator):
    _estimator_type = "forecaster"

    def __init__(self):
        self._oh = None  # observation horizon, i.e. time points seen in fit or update
        self._now = None  # time point in observation horizon now which to make forecasts
        self._is_fitted = False
        self._fh = None
        super(BaseForecaster, self).__init__()

    @property
    def is_fitted(self):
        """Has `fit` been called?"""
        return self._is_fitted

    def _check_is_fitted(self):
        """Check if the forecaster has been fitted.

        Raises
        ------
        NotFittedError
            if the forecaster has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(f"This instance of {self.__class__.__name__} has not "
                                 f"been fitted yet; please call `fit` first.")

    @property
    def oh(self):
        """The observation horizon, i.e. the seen data
        passed either to `fit` or one of the `update` methods.

        Returns
        -------
        oh : pd.Series
            The available observation horizon
        """
        return self._oh

    def _set_oh(self, y):
        """Set and update the observation horizon

        Parameters
        ----------
        y : pd.Series
        """
        # input checks
        oh = check_y(y)

        # for updating: append observation horizon to previous one
        if self.is_fitted:
            # update observation horizon, merging both series on time index,
            # overwriting old data with new data
            self._oh = oh.combine_first(self.oh)

        # for fitting: since no previous observation horizon is present, set new one
        else:
            self._oh = oh

        # by default, set now to the end of the observation horizon
        self._set_now(oh.index[-1])

    @property
    def now(self):
        """Now, the time point at which to make forecasts.

        Returns
        -------
        now : int
        """
        return self._now

    def _set_now(self, now):
        """Set and update now, the time point at which to make forecasts.

        Parameters
        ----------
        now : int
        """
        self._now = now

    @property
    def fh(self):
        """The forecasting horizon"""
        # raise error if some method tries to accessed it before it has been set
        if self._fh is None:
            raise ValueError("No `fh` has been set yet.")
        return self._fh

    def fit(self, y_train, fh=None, X_train=None):
        raise NotImplementedError()

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        raise NotImplementedError("abstract method")

    def update(self, y_new, X_new=None, update_params=False):
        raise NotImplementedError()

    def update_predict(self, y_test, cv=None, X_test=None, update_params=False, return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
        """Make predictions and updates iteratively over the test set.

        Parameters
        ----------
        y_test : pd.Series
        cv : cross-validation generator, optional (default=None)
        X_test : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=False)
        return_pred_int : bool, optional (default=False)
        alpha : int or list of ints, optional (default=None)

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
        """

        # input checks
        if X_test is not None or return_pred_int:
            raise NotImplementedError()

        self._check_is_fitted()

        # keep previous now to reset after
        # update predict routine
        previous_now = self.now

        if cv is None:
            # if no cv is provided, use default
            cv = SlidingWindowSplitter(fh=self.fh)
        else:
            # otherwise check provided cv
            cv = check_cv(cv)

        fh = check_fh(cv.fh)

        #  update oh, but reset now to time point before new data
        self._set_oh(y_test)
        oh = self.oh
        self._set_now(y_test.index[0] - 1)

        #  get window length from cv
        window_length = cv.window_length

        # if any window would be before the first observation of the observation horizon,
        # oh with missing values
        oh_start = self.oh[0]
        start = self.now - window_length + 1
        is_before_oh = start < oh_start
        if is_before_oh:
            index = np.arange(self.now - window_length, self.now) + 1
            presample = pd.Series(np.full(window_length, np.nan), index=index)
            oh = presample.append(oh)

        # select subset to iterate over from observation horizon
        y = oh.iloc[start:]
        time_index = y.index

        # allocate lists for prediction results
        y_preds = []
        nows = []  # time points at which we predict

        # iteratively call update and predict, first update will contain only the
        # last window before the given data and no new data, so that we start by
        # predicting the first value of the given data
        for new_window, _ in cv.split(time_index):
            y_new = y.iloc[new_window]

            # if presample, only update now and predict nan
            now = y_new.index[-1]
            is_presample = now - window_length + 1 < oh_start
            if is_presample:
                self._set_now(now)
                index = self._get_absolute_fh(fh)
                y_test_pred = pd.Series(np.full(len(fh), np.nan), index=index)

            # otherwise, run update and predict
            else:
                # update: while the observation horizon is already_test updated, we still need to
                # update `now` and may_test want to update fitted parameters
                self.update(y_new, update_params=update_params)

                # predict: make a forecast at each step
                y_test_pred = self.predict(fh, X=X_test, return_pred_int=return_pred_int, alpha=alpha)

            y_preds.append(y_test_pred)
            nows.append(self.now)

        # reset now
        self._set_now(previous_now)

        # format results
        if len(fh) > 1:
            # return data frame when we predict multiple steps ahead
            y_preds = pd.DataFrame(y_preds).T
            y_preds.columns = nows
        else:
            # return series for single step ahead predictions
            y_preds = pd.concat(y_preds)
        return y_preds

    def update_predict_single(self, y_new, fh=None, X=None, update_params=False, return_pred_int=False,
                              alpha=DEFAULT_ALPHA):
        """Allows for more efficient update-predict routines than calling them sequentially"""
        # when nowcasting, X may be longer than y, X must be cut to same length as y so that same time points are
        # passed to update, the remaining time points of X are passed to predict
        if X is not None or return_pred_int:
            raise NotImplementedError()

        self.update(y_new, X_new=X, update_params=update_params)
        return self.predict(fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha)

    def compute_pred_errors(self, alpha=DEFAULT_ALPHA):
        """
        Prediction errors. If alpha is iterable, errors will be calculated for
        multiple confidence levels.
        """
        raise NotImplementedError()

    def compute_pred_int(self, y_pred, alpha=DEFAULT_ALPHA):
        """
        Get the prediction intervals for the forecast. If alpha is iterable, multiple
        intervals will be calculated.
        """
        errors = self.compute_pred_errors(alpha=alpha)

        # for multiple alphas, errors come in a list;
        # for single alpha, they come as a single pd.Series,
        # wrap it here into a list to make it iterable,
        # to avoid code duplication
        if isinstance(errors, pd.Series):
            errors = [errors]

        # compute prediction intervals
        pred_int = [
            pd.DataFrame({
                "lower": y_pred - error,
                "upper": y_pred + error
            })
            for error in errors
        ]

        # for a single alpha, return single pd.DataFrame
        if len(pred_int) == 1:
            return pred_int[0]

        # otherwise return list of pd.DataFrames
        return pred_int

    def score(self, y_test, fh=None, X=None):
        """
        Returns the SMAPE on the given forecast horizon.
        Parameters
        ----------
        y_test : pandas.Series
            Target time series to which to compare the forecasts.
        fh : int or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.
        Returns
        -------
        score : float
            SMAPE score of self.predict(fh=fh, X=X) with respect to y.
        See Also
        --------
        :meth:`sktime.performance_metrics.forecasting.smape_loss`.`
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        return smape_loss(y_test, self.predict(fh=fh, X=X))

    def plot(self, fh=None, alpha=(0.05, 0.2), y_train=None, y_test=None, y_pred=None, fig=None, ax=None, title=None,
             score='lower right', **kwargs):
        """Plot a forecast.

        Parameters
        ----------
        fh : int or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        alpha : float or array-like, optional (default=(0.05, 0.2))
            Alpha values for a confidence level or list of confidence levels to plot
            prediction intervals for.
        y_train : :class:`pandas.Series`, optional
            The original training data to plot alongside the forecast.
        y_test : :class:`pandas.Series`, optional
            The actual data to compare to the forecast for in-sample forecasts
            ("nowcasts").
        y_pred : :class:`pandas.Series`, optional
            Previously calculated forecast from the same forecaster. If omitted, a
            forecast will be generated automatically using :meth:`.predict()`.
        fig : :class:`matplotlib.figure.Figure`, optional
            A figure to plot the graphic on.
        ax : :class:`matplotlib.axes.Axes`, optional
            The axis on which to plot the graphic. If not provided, a new one
            will be created.
        title : str
            Title of plot
        score : str, optional (default="lower right")
            Where to draw a text box showing the score of the forecast if possible.
            If set to None, no score will be displayed.
        kwargs
            Additional keyword arguments to pass to :meth:`.predict`.
        Returns
        -------
        ax : :class:`matplotlib.axes.Axes`
            The axis on which the graphic was drawn.
        """

        self._set_fh(fh)

        if y_pred is None:
            y_pred = self.predict(fh=self.fh, **kwargs)

        y_pred_label = y_pred.name if y_pred.name else f"Forecast ($h = {len(self.fh)}$)"

        # Import dynamically to avoid creating matplotlib dependencies.
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.patches import Patch

        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.gca()

        if title:
            ax.set_title(title)

        y_col = None
        if y_train is not None:
            label = f"{y_train.name} (Train)" if y_train.name else "Train"
            y_train.plot(ax=ax, label=label)
            y_col = ax.get_lines()[-1].get_color()

        if y_test is not None:
            label = f"{y_test.name} (Test)" if y_test.name else "Test"
            dense_dots = (0, (1, 1))
            y_test.plot(ax=ax, label=label, ls=dense_dots, c=y_col)

        y_pred.plot(ax=ax, label=y_pred_label)
        fcast_col = ax.get_lines()[-1].get_color()

        if score and y_test is not None and y_train is not None:
            try:
                y_score = self.score(y_test, fh=self.fh, X=kwargs.get("X"))
                text_box = AnchoredText(
                    f"Score = ${y_score:.3f}$", frameon=True, loc=score
                )
                ax.add_artist(text_box)
            except ValueError:
                # Cannot calculate score if y_test and fh indices don't align.
                pass

        axhandles, axlabels = ax.get_legend_handles_labels()
        if alpha is not None:
            # Plot prediction intervals if available.
            try:
                if isinstance(alpha, (np.integer, np.float)):
                    alpha = [alpha]

                # trans = np.linspace(0.25, 0.65, num=len(alpha), endpoint=False)
                transp = 0.25
                # Plot widest intervals first.
                alpha = sorted(alpha)

                last_transp = 0
                for al in alpha:
                    intvl = self.compute_pred_int(y_pred=y_pred, alpha=al)
                    ax.fill_between(
                        y_pred.index,
                        intvl.upper,
                        intvl.lower,
                        fc=fcast_col,
                        ec=fcast_col,
                        alpha=transp,
                        lw=0
                    )

                    # Each level gives an effective transparency through overlapping.
                    # Reflect this in the legend.
                    effective_transp = composite_alpha(last_transp, transp)
                    axhandles.append(Patch(fc=fcast_col, alpha=effective_transp, ec=fcast_col))
                    last_transp = effective_transp

                    axlabels.append(f"{round((1 - al) * 100)}% conf")

            except NotImplementedError:
                pass

        ax.legend(handles=axhandles, labels=axlabels)

        if fig is not None:
            fig.tight_layout()

        return ax

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Abstract base method, implemented by mixin classes.

        Parameters
        ----------
        fh : None, int, list, np.array
        """
        #
        raise NotImplementedError()

    def _get_absolute_fh(self, fh=None):
        """Convert the user-defined forecasting horizon relative to the end
        of the observation horizon into the absolute time index.

        Returns
        -------
        fh : np.array
            The absolute time index of the forecasting horizon
        """
        # user defined forecasting horizon `fh` is relative to the end of the
        # observation horizon, i.e. `now`
        if fh is None:
            fh = self.fh
        fh_abs = self.now + fh

        # for in-sample predictions, check if forecasting horizon is still within
        # observation horizon
        if any(fh_abs < 0):
            raise ValueError("Forecasting horizon `fh` includes time points "
                             "before observation horizon")
        return np.sort(fh_abs)

    def _get_index_fh(self, fh=None):
        """Convert the step-ahead forecast horizon relative to the end
        of the observation horizon into the zero-based forecasting horizon
        for array indexing.
        Returns
        -------
        fh : np.array
            The zero-based index of the forecasting horizon
        """
        if fh is None:
            fh = self.fh
        return fh - 1

    @contextmanager
    def _detached_now(self):
        """context manager to detach now"""
        now = self.now  # remember initial now
        try:
            yield
        finally:
            # re-set now to initial state
            self._set_now(now)


class OptionalForecastingHorizonMixin:
    """Mixin class for forecasters which can take the forecasting horizon either
    during fitting or prediction."""

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Parameters
        ----------
        fh : None, int, list or np.ndarray
        """
        if hasattr(self, "is_fitted"):
            is_fitted = self.is_fitted
        else:
            raise AttributeError("No `is_fitted` attribute found")

        if fh is None:
            if is_fitted:
                # if no fh passed and there is none already, raise error
                if self._fh is None:
                    raise ValueError("The forecasting horizon `fh` must be passed either to `fit` or `predict`, "
                                     "but was found in neither.")
                # otherwise if no fh passed, but there is one already, we can simply use that one
        else:
            # if fh is passed, validate first, then check if there is one already,
            # and overwrite with appropriate warning
            fh = check_fh(fh)
            if is_fitted:
                # raise warning if existing fh and new one don't match
                if self._fh is not None and not np.array_equal(fh, self._fh):
                    warn("The provided forecasting horizon `fh` is different from the "
                         "previously provided one; the new one will be used.")
            self._fh = fh


class RequiredForecastingHorizonMixin:
    """Mixin class for forecasters which require the forecasting horizon during fitting."""

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Parameters
        ----------
        fh : None, int, list, np.ndarray
        """

        msg = f"This is because fitting of the `{self.__class__.__name__}` depends on `fh`. "

        if hasattr(self, "is_fitted"):
            is_fitted = self.is_fitted
        else:
            raise AttributeError("No `is_fitted` attribute found")

        if fh is None:
            if is_fitted:
                # intended workflow, no fh is passed when the forecaster is already fitted
                pass
            else:
                # fh must be passed when forecaster is not fitted yet
                raise ValueError("The forecasting horizon `fh` must be passed to `fit`, "
                                 "but none was found. " + msg)
        else:
            fh = check_fh(fh)
            if is_fitted:
                if not np.array_equal(fh, self._fh):
                    # raise error if existing fh and new one don't match
                    raise ValueError(
                        f"A different forecasting horizon `fh` has been provided from "
                        f"the one seen in `fit`. If you want to change the forecasting "
                        f"horizon, please re-fit the forecaster. " + msg)
                # if existing one and new match, ignore new one
                pass
            else:
                # intended workflow: fh is passed when forecaster is not fitted yet
                self._fh = fh


class BaseLastWindowForecaster(BaseForecaster):

    def __init__(self):
        super(BaseLastWindowForecaster, self).__init__()
        self._window_length = None

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        self._check_is_fitted()
        self._set_fh(fh)

        # distinguish between in-sample and out-of-sample prediction
        is_in_sample = self.fh <= 0
        is_out_of_sample = np.logical_not(is_in_sample)

        # pure out-of-sample prediction
        if np.all(is_out_of_sample):
            return self._predict_out_of_sample(self.fh, X=X, return_pred_int=return_pred_int, alpha=DEFAULT_ALPHA)

        # pure in-sample prediction
        elif np.all(is_in_sample):
            return self._predict_in_sample(self.fh, X=X, return_pred_int=return_pred_int, alpha=DEFAULT_ALPHA)

        # mixed in-sample and out-of-sample prediction
        else:
            fh_in_sample = self.fh[is_in_sample]
            fh_out_of_sample = self.fh[is_out_of_sample]

            y_pred_in = self._predict_in_sample(fh_in_sample, X=X, return_pred_int=return_pred_int,
                                                alpha=DEFAULT_ALPHA)
            y_pred_out = self._predict_out_of_sample(fh_out_of_sample, X=X, return_pred_int=return_pred_int,
                                                     alpha=DEFAULT_ALPHA)
            return y_pred_in.append(y_pred_out)

    def update_predict(self, y_test, cv=None, X_test=None, update_params=False, return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
        self._check_is_fitted()
        self._set_oh(y_test)

        # if no cv is provided, use default, otherwise check provided cv
        cv = check_cv(cv) if cv is not None else SlidingWindowSplitter(fh=self.fh)

        return self._predict_moving_cutoff(y_test, cv, update=True, update_params=update_params,
                                           return_pred_int=return_pred_int, alpha=alpha)

    def _predict_out_of_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        assert all(fh > 0)
        return self._predict_fixed_cutoff(fh, X=X, return_pred_int=return_pred_int, alpha=alpha)

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        assert all(fh <= 0)
        #  convert in-sample fh steps to cutoff points
        index = self._get_absolute_fh(fh)
        cutoffs = index - 1  # points before fh steps
        cv = ManualWindowSplitter(cutoffs, fh=1, window_length=self._window_length)
        y_train = self.oh
        return self._predict_moving_cutoff(y_train, cv, X_test=X, update=False, return_pred_int=return_pred_int,
                                           alpha=alpha)

    def _predict_fixed_cutoff(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int or X is not None:
            raise NotImplementedError()

        last_window = self._get_last_window()
        if len(last_window) == 0:
            #  pre-sample time points will return empty last window
            #  predict nan
            y_pred = self._predict_nan(fh, return_pred_int=return_pred_int)
        else:
            y_pred = self._predict(last_window, fh, return_pred_int=return_pred_int, alpha=alpha)
        index = self._get_absolute_fh(fh)
        return pd.Series(y_pred, index=index)

    def _predict_moving_cutoff(self, y_test, cv, X_test=None, update=True, update_params=False, return_pred_int=False,
                               alpha=DEFAULT_ALPHA):
        """static moving cutoff predictions, i.e. no previously
        predicted values are used to make subsequent predictions"""
        if not update and update_params:
            raise ValueError("`update_params` can only be used if `update`=True")

        fh = cv.fh
        window_length = cv.window_length

        with self._detached_now():
            # set before new data, so that first prediction is
            # first observation in new data
            self._set_now(y_test.index[0] - 1)
            start = self.now - window_length + 1

            # extend observation horizon into the past if any window
            # would be before the first observation
            start_oh = self.oh.index[0]
            is_pre_sample = start_oh > start
            if is_pre_sample:
                index = np.arange(self.now - window_length, self.now) + 1
                presample = pd.Series(np.full(window_length, np.nan), index=index)
                self._set_oh(presample)
                y = self.oh
            else:
                y = self.oh.iloc[start:]

            # initialise lists
            y_preds = []
            nows = []

            # iterate over data
            for i, (new_window, _) in enumerate(cv.split(y)):
                y_new = y.iloc[new_window]

                # if udpate=True, run full update, otherwise only update now
                if update:
                    self.update(y_new, update_params=update_params)
                else:
                    self._set_now(y_new.index[-1])

                y_pred = self._predict_fixed_cutoff(fh, X=X_test, return_pred_int=return_pred_int, alpha=alpha)
                y_preds.append(y_pred)
                nows.append(self.now)

            if len(fh) == 1:
                # return series for single step ahead predictions
                y_pred = pd.concat(y_preds)
            else:
                # return data frame when we predict multiple steps ahead
                y_pred = pd.DataFrame(y_preds).T
                y_pred.columns = nows

            return y_pred

    def _predict(self, last_window, fh, return_pred_int=False, alpha=DEFAULT_ALPHA):
        raise NotImplementedError("abstract method")

    def _predict_nan(self, fh, return_pred_int=False):
        if return_pred_int:
            raise NotImplementedError()
        return np.full(len(fh), np.nan)

    def _get_last_window(self):
        start = self.now - self._window_length + 1
        end = self.now
        return self.oh.loc[start:end].values

