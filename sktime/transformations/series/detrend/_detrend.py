#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformations to detrend a time series."""

__all__ = ["Detrender"]
__author__ = ["mloning", "SveaMeyer13", "KishManani", "fkiraly"]

import pandas as pd

from sktime.datatypes import update_data
from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.base import BaseTransformer


class Detrender(BaseTransformer):
    """Remove a :term:`trend <Trend>` from a series.

    This transformer uses any forecaster and returns the in-sample residuals
    of the forecaster's predicted values.

    The Detrender works as follows:
    in "fit", the forecaster is fit to the input data, i.e., ``forecaster.fit(y=X)``.
    in "transform", returns forecast residuals of forecasts at the data index.
    That is, ``transform(X)`` returns ``X - forecaster.predict(fh=X.index)`` (additive)
    or ``X / forecaster.predict(fh=X.index)`` (multiplicative detrending).
    Depending on time indices, this can generate in-sample or out-of-sample residuals.

    For example, to remove the linear trend of a time series:
        forecaster = PolynomialTrendForecaster(degree=1)
        transformer = Detrender(forecaster=forecaster)
        yt = transformer.fit_transform(y_train)

    The detrender can also be used in a pipeline for residual boosting,
    by first detrending and then fitting another forecaster on residuals.

    Parameters
    ----------
    forecaster : sktime forecaster, follows BaseForecaster, default = None.
        The forecasting model to remove the trend with
            (e.g. PolynomialTrendForecaster).
        If forecaster is None, PolynomialTrendForecaster(degree=1) is used.
        Must be a forecaster to which ``fh`` can be passed in ``predict``.
    model : {"additive", "multiplicative"}, default="additive"
        If ``model="additive"`` the ``forecaster.transform`` subtracts the trend,
        i.e., ``transform(X)`` returns ``X - forecaster.predict(fh=X.index)``
        If ``model="multiplicative"`` the ``forecaster.transform`` divides by the trend,
        i.e., ``transform(X)`` returns ``X / forecaster.predict(fh=X.index)``

    Attributes
    ----------
    forecaster_ : Fitted forecaster
        Forecaster that defines the trend in the series.

    See Also
    --------
    Deseasonalizer
    STLTransformer

    Examples
    --------
    >>> from sktime.transformations.series.detrend import Detrender
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Detrender(forecaster=PolynomialTrendForecaster(degree=1))
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["mloning", "SveaMeyer13", "KishManani", "fkiraly"],
        "maintainers": ["SveaMeyer13", "KishManani"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": False,
        "capability:inverse_transform": True,
        "transform-returns-same-time-index": True,
    }

    def __init__(self, forecaster=None, model="additive"):
        self.forecaster = forecaster
        self.model = model

        super().__init__()

        # default for forecaster - written to forecaster_ to not overwrite param
        if self.forecaster is None:
            self.forecaster_ = PolynomialTrendForecaster(degree=1)
        else:
            self.forecaster_ = forecaster.clone()

        allowed_models = ("additive", "multiplicative")
        if model not in allowed_models:
            raise ValueError("`model` must be 'additive' or 'multiplicative'")

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to fit transform to
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        if not self.forecaster_.get_tag("requires-fh-in-fit", True):
            self.forecaster_.fit(y=X, X=y)
        else:
            self._X = X
            self._y = y
        return self

    def _get_fh_from_X(self, X):
        """Obtain fh from X, which can be simple or hierarchical."""
        if not isinstance(X.index, pd.MultiIndex):
            time_index = X.index
        else:
            time_index = X.index.get_level_values(-1).unique()
        return ForecastingHorizon(time_index, is_relative=False)

    def _get_fitted_forecaster(self, X, y, fh):
        """Obtain fitted forecaster from self."""
        if self.forecaster_.get_tag("requires-fh-in-fit", True):
            X = update_data(self._X, X)
            y = update_data(self._y, y)
            forecaster = self.forecaster_.clone().fit(y=X, X=y, fh=fh)
        else:
            forecaster = self.forecaster_
        return forecaster

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X, detrended series
        """
        fh = self._get_fh_from_X(X=X)
        forecaster = self._get_fitted_forecaster(X=X, y=y, fh=fh)

        X_pred = forecaster.predict(fh=fh, X=y)

        if self.model == "additive":
            return X - X_pred
        elif self.model == "multiplicative":
            return X / X_pred

    def _inverse_transform(self, X, y=None):
        """Logic used by ``inverse_transform`` to reverse transformation on ``X``.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be inverse transformed
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            inverse transformed version of X
        """
        fh = self._get_fh_from_X(X=X)
        # we pass X and y as None, since the X passed is inverse transformed (detrended)
        # the fit, in case fh needs be passed late, is done on remembered data from fit
        forecaster = self._get_fitted_forecaster(X=None, y=None, fh=fh)

        X_pred = forecaster.predict(fh=fh, X=y)

        if self.model == "additive":
            return X + X_pred
        elif self.model == "multiplicative":
            return X * X_pred

    def _update(self, X, y=None, update_params=True):
        """Update the parameters of the detrending estimator with new data.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to fit transform to
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation
        update_params : bool, default=True
            whether the model is updated. Yes if true, if false, simply skips call.
            argument exists for compatibility with forecasting module.

        Returns
        -------
        self : an instance of self
        """
        if not self.forecaster_.get_tag("requires-fh-in-fit", True):
            self.forecaster_.update(y=X, X=y, update_params=update_params)
        else:
            self._X = update_data(self._X, X)
            self._y = update_data(self._y, y)
        return self

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.forecasting.trend import TrendForecaster

        params1 = {"forecaster": TrendForecaster()}
        params2 = {"model": "multiplicative"}

        return [params1, params2]
