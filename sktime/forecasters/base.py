__all__ = ["BaseForecaster", "BaseSingleSeriesForecaster", "BaseUpdateableForecaster"]
__author__ = ["Markus Löning"]


from warnings import warn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted

from sktime.utils.validation.forecasting import check_conf_level
from sktime.utils.validation.forecasting import validate_fh
from sktime.utils.validation.forecasting import validate_X
from sktime.utils.validation.forecasting import validate_y
from sktime.utils.validation.forecasting import validate_y_X
from sktime.utils.data_container import tabularise


# Default confidence level for prediction intervals.
DEFAULT_CLVL = 0.95


class BaseForecaster(BaseEstimator):
    """
    Base class for forecasters.
    """
    _estimator_type = "forecaster"

    def __init__(self):
        self._time_index = None  # forecasters need to keep track of time index of target series
        self._is_fitted = False
        self._fh = None

    def fit(self, y, fh=None, X=None):
        """
        Fit forecaster.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        fh : int or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pandas.DataFrame, shape=[n_instances, n_columns], optional (default=None)
            An optional 2-d dataframe of exogenous variables.

        Returns
        -------
        self : returns an instance of self.
        """
        # check input
        validate_y_X(y, X)

        # validate forecasting horizon if passed
        if fh is not None:
            fh = validate_fh(fh)
            self._fh = fh

        # Keep index for predicting where forecasters horizon will be relative to y seen in fit
        self._time_index = y.index

        # Make interface compatible with estimators that only take y and no X
        kwargs = {} if X is None else {'X': X}

        # Internal fit.
        self._fit(y, fh=fh, **kwargs)
        self._is_fitted = True
        return self

    def _fit(self, fh=None, **kwargs):
        """Internal fit implemented by specific forecasters"""
        raise NotImplementedError()

    def _prepare_fh(self, fh):
        """
        Shared code for preparing the forecasting horizon for fit, predict and
        predict_intervals.
        """
        # validate forecasting horizon
        # if no fh is passed to predict, check if it was passed to fit; if so, use it;
        # otherwise raise error
        if fh is None:
            if self._fh is not None:
                fh = self._fh
            else:
                raise ValueError("Forecasting horizon (fh) must be passed to `fit` or `predict`")

        # if fh is passed to predict, check if fh was also passed to fit; if so, check
        # if they are the same; if not, raise warning
        else:
            fh = validate_fh(fh)
            if self._fh is not None:
                if not np.array_equal(fh, self._fh):
                    warn("The forecasting horizon (fh) passed to `predict` is different "
                         "from the fh passed to `fit`")
            self._fh = fh  # use passed fh; overwrites fh if it was passed to fit already

        return fh

    def predict(
        self,
        fh=None,
        X=None,
        levels=None,
        **kwargs
    ):
        """
        Predict using fitted estimator.

        Parameters
        ----------

        fh : int or array-like, optional (default=1)
            The forecasters horizon with the steps ahead to to predict. Default is
            one-step ahead forecast, i.e. np.array([1])

        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.

        levels : float or list-like, optional
            A confidence level expressed as a fraction to provide prediction errors
            for. If this is set, the return value will be a tuple of (predictions,
            prediction_errors). A list of multiple confidence levels may be provided
            which will return a list of prediction errors as the second item in the
            tuple. See :meth:`.prediction_errors` for more info on prediction errors.

        kwargs : keyword arguments
            Any other options that may be taken by the forecaster's :meth:`_predict`
            method.

        Returns
        -------

        predictions : pandas.Series, shape=(len(fh),)
            series of predicted values.

        (predictions, prediction_errors)
            If confidence level(s) were supplied.
        """
        # check input
        check_is_fitted(self, '_is_fitted')
        validate_X(X)

        fh = self._prepare_fh(fh)

        # make interface compatible with estimators that only take y
        kwargs = {} if X is None else {'X': X}

        # estimator specific implementation of fit method
        pred = self._predict(fh=fh, **kwargs)

        if levels is not None:
            if isinstance(levels, (int, float)):
                level = levels
                errs = self.prediction_errors(fh=fh, conf_lvl=level)
            else:
                errs = [
                    self.prediction_errors(fh=fh, conf_lvl=level) for level in levels
                ]

            return pred, errs

        return pred

    def _predict(self, fh=None, **kwargs):
        """Internal predict implemented by specific forecasters.
        """
        raise NotImplementedError()

    def prediction_errors(self, fh=None, conf_lvl=DEFAULT_CLVL):
        """
        Calculate the prediction errors for the given forecast horizon.

        Prediction intervals may be calculated by adding/subtracting these errors from
        the predictions for the same forecast horizon.

        Parameters
        ----------

        fh : int or array-like, optional (default=1)
            The forecast horizon with the steps ahead to calculate intervals for.
            Default is one-step ahead forecast.

        conf_lvl : float
            The confidence level to use for the errors. Must be within the open
            interval (0.0, 1.0).
        """
        check_conf_level(conf_lvl)
        fh = self._prepare_fh(fh)

        return self._prediction_errors(fh=fh, conf_lvl=conf_lvl)

    def _prediction_errors(self, fh=None, conf_lvl=DEFAULT_CLVL):
        raise NotImplementedError()

    def plot(
        self,
        *,
        fh=None,
        conf_lvls=(0.95, 0.8),
        y_train=None,
        y_true=None,
        fig=None,
        ax=None,
        score='lower right',
        **kwargs,
    ):
        """
        Plot a forecast.

        Parameters
        ----------

        fh : int or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        conf_lvls : float or array-like, optional (default=(0.95, 0.8))
            A confidence level or list of confidence levels to plot prediction
            intervals for.

        y_train : :class:`pandas.Series`, optional
            The original training data to plot alongside the forecast.

        y_true : :class:`pandas.Series`
            The actual data to compare to the forecast for in-sample forecasts
            ("nowcasts").

        fig : :class:`matplotlib.figure.Figure`, optional
            A figure to plot the graphic on.

        ax : :class:`matplotlib.axes.Axes`, optional
            The axis on which to plot the graphic. If not provided, a new one
            will be created.

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

        y_hat = self.predict(fh=fh, **kwargs)
        y_hat.name = f"Forecast ($h = {len(fh)}$)"

        # Import dynamically to avoid creating matplotlib dependencies.
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.patches import Patch

        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.gca()

        if y_train is not None:
            y_train.plot(ax=ax)

        if y_true is not None:
            y_true.plot(ax=ax)

        y_hat.plot(ax=ax, ls="--")
        y_hat_line = ax.get_lines()[-1]

        if score:
            try:
                y_score = self.score(y_true=y_true, fh=fh, X=kwargs.get("X"))
                text_box = AnchoredText(
                    f"Score = ${y_score:.3f}$", frameon=True, loc=score
                )
                ax.add_artist(text_box)
            except ValueError:
                # Cannot calculate score if y_true and fh indices don't align.
                pass

        axhandles, axlabels = ax.get_legend_handles_labels()
        if conf_lvls is not None:
            # Plot prediction intervals if available.
            try:
                if isinstance(conf_lvls, (int, float)):
                    conf_lvls = [conf_lvls]

                col = y_hat_line.get_color()
                alphas = np.linspace(0.15, 0.65, num=len(conf_lvls), endpoint=False)
                # Plot widest intervals first.
                conf_lvls = list(reversed(sorted(conf_lvls)))

                for alpha, lvl in zip(alphas, conf_lvls):
                    err = self.prediction_errors(fh=fh, conf_lvl=lvl)
                    ax.fill_between(
                        y_hat.index, y_hat+err, y_hat-err, fc=col, alpha=alpha, lw=0
                    )
                    axhandles.append(Patch(fc=col, alpha=alpha, ec=col))
                    axlabels.append(f"{round(lvl*100)}% conf")

            except NotImplementedError:
                pass

        ax.legend(handles=axhandles, labels=axlabels)

        if fig is not None:
            fig.tight_layout()

        return ax

    def score(self, y_true, fh=None, X=None):
        """
        Returns the root mean squared error on the given forecast horizon.

        Parameters
        ----------
        y_true : pandas.Series
            Target time series to which to compare the forecasts.
        fh : int or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.

        Returns
        -------
        score : float
            Mean squared error of self.predict(fh=fh, X=X) with respect to y.
        """
        # only check y here, X and fh will be checked during predict
        validate_y(y_true)

        # Predict y_pred
        # pass exogenous variable to predict only if given, as some forecasters may not accept X in predict
        kwargs = {} if X is None else {'X': X}
        y_pred = self.predict(fh=fh, **kwargs)

        # Check if passed true time series coincides with forecast horizon of predicted values
        if not y_true.index.equals(y_pred.index):
            raise ValueError(f"Index of passed time series `y_true` does not match index of predicted time series; "
                             f"make sure the forecasters horizon `fh` matches the time index of `y_true`")

        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def _get_y_index(y):
        """Helper function to get (time) index of y used in fitting for later comparison
        with forecast horizon
        """
        y = y.iloc[0]
        index = y.index if hasattr(y, 'index') else pd.RangeIndex(len(y))
        return index

    @staticmethod
    def _prepare_X(X):
        """Helper function to transform nested pandas DataFrame X into 2d numpy array as required by `statsmodels`
        estimators.

        Parameters
        ----------
        X : pandas.DataFrame, shape=[1, n_variables]
            Nested dataframe with series of shape [n_obs,] in cells

        Returns
        -------
        Xt : ndarray, shape=[n_obs, n_variables]
        """
        if X is None:
            return X

        if X.shape[1] > 1:
            Xl = X.iloc[0, :].tolist()
            Xt = np.column_stack(Xl)
        else:
            Xt = tabularise(X).values.T

        return Xt


class BaseUpdateableForecaster(BaseForecaster):
    # TODO should that be a mixin class instead?
    """
    Base class for forecasters with update functionality.
    """

    def __init__(self):
        super(BaseUpdateableForecaster, self).__init__()
        self._is_updated = False

    def update(self, y, X=None):
        """
        Update forecasts using new data via Kalman smoothing/filtering of
        forecasts obtained from previously fitted forecaster.

        Parameters
        ----------
        y : pandas.Series
            Updated time series which to use for updating the previously fitted forecaster.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must also be provided
            exogenous features for making predictions.

        Returns
        -------
        self : An instance of self
        """
        # check inputs
        check_is_fitted(self, '_is_fitted')
        validate_y_X(y, X)
        self._validate_y_update(y)

        self._update(y, X=X)
        self._is_updated = True
        return self

    def _validate_y_update(self, y):
        """
        Helper function to check the ``y`` passed to update the estimator
        """
        # TODO add input checks for X when updating
        # TODO add additional input checks for update data, i.e. that update data is newer than data seen in fit
        if not isinstance(y.index, type(self._time_index)):
            raise ValueError("The time index of the target series (y) does not match"
                             " the time index of y seen in `fit`")


class BaseSingleSeriesForecaster(BaseForecaster):
    """Statsmodels interface wrapper class, classical forecaster which implements predict method for single-series/univariate fitted/updated classical
    forecasters techniques without exogenous variables (X).
    """

    def _predict(self, fh=None):
        """
        Internal predict.

        Parameters
        ----------
        fh : int or array-like, optional (default=1)
            The forecasters horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """

        # Convert step-ahead prediction horizon into zero-based index
        fh_idx = fh - np.min(fh)

        # Predict fitted model with start and end points relative to start of train series
        fh = len(self._time_index) - 1 + fh
        start = fh[0]
        end = fh[-1]
        y_pred = self._fitted_estimator.predict(start=start, end=end)

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        return y_pred.iloc[fh_idx]
