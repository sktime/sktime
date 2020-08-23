#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["TransformedTargetForecaster"]

from sklearn.base import clone
from sktime.base import BaseHeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import MetaForecasterMixin
from sktime.forecasting.base._sktime import BaseSktimeForecaster
from sktime.forecasting.base._sktime import OptionalForecastingHorizonMixin
from sktime.transformers.single_series.base import BaseSingleSeriesTransformer
from sktime.utils.validation.forecasting import check_y


class TransformedTargetForecaster(MetaForecasterMixin,
                                  OptionalForecastingHorizonMixin,
                                  BaseSktimeForecaster,
                                  BaseHeterogenousMetaEstimator):
    """Meta-estimator for forecasting transformed time series."""

    _required_parameters = ["steps"]

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = None
        super(TransformedTargetForecaster, self).__init__()

    def _check_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._check_names(names)

        # validate estimators
        transformers = estimators[:-1]
        forecaster = estimators[-1]

        allowed_transformer_type = BaseSingleSeriesTransformer
        for t in transformers:
            # Transformers must be endog/exog transformers
            if not isinstance(t, allowed_transformer_type):
                raise TypeError(f"All intermediate steps should be "
                                f"instances of {allowed_transformer_type}, "
                                f"but "
                                f"transformer: {t} is not.")

        allowed_forecaster_type = BaseForecaster
        if not isinstance(forecaster, allowed_forecaster_type):
            raise TypeError(
                f"Last step of {self.__class__.__name__} must be of type: "
                f"{allowed_forecaster_type}, "
                f"but forecaster: {forecaster} is not.")

        # Shallow copy
        return list(self.steps)

    def _iter_transformers(self, reverse=False):

        # exclude final forecaster
        steps = self.steps_[:-1]

        if reverse:
            steps = reversed(steps)

        for idx, (name, transformer) in enumerate(steps):
            yield idx, name, transformer

    def __len__(self):
        """
        Returns the length of the Pipeline
        """
        return len(self.steps)

    @property
    def named_steps(self):
        """Map the steps to a dictionary"""
        return dict(self.steps)

    def fit(self, y_train, fh=None, X_train=None):
        """Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X_train : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self.steps_ = self._check_steps()
        self._set_y_X(y_train, X_train)
        self._set_fh(fh)

        # transform
        yt = check_y(y_train)
        for step_idx, name, transformer in self._iter_transformers():
            t = clone(transformer)
            yt = t.fit_transform(yt)
            self.steps_[step_idx] = (name, t)

        # fit forecaster
        name, forecaster = self.steps[-1]
        f = clone(forecaster)
        f.fit(yt, fh=fh, X_train=X_train)
        self.steps_[-1] = (name, f)

        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None, return_pred_int=False,
                 alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()

        forecaster = self.steps_[-1][1]
        y_pred = forecaster.predict(fh=fh, X=X,
                                    return_pred_int=return_pred_int,
                                    alpha=alpha)

        for step_idx, name, transformer in self._iter_transformers(
                reverse=True):
            y_pred = transformer.inverse_transform(y_pred)

        return y_pred

    def update(self, y_new, X_new=None, update_params=False):
        """Update fitted paramters

        Parameters
        ----------
        y_new : pd.Series
        X_new : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """
        self.check_is_fitted()
        self._update_y_X(y_new, X_new)

        for step_idx, name, transformer in self._iter_transformers():
            if hasattr(transformer, "update"):
                transformer.update(y_new, update_params=update_params)
                self.steps_[step_idx] = (name, transformer)

        name, forecaster = self.steps_[-1]
        forecaster.update(y_new, update_params=update_params)
        self.steps_[-1] = (name, forecaster)
        return self

    def transform(self, y):
        self.check_is_fitted()
        yt = check_y(y)
        for step_idx, name, transformer in self._iter_transformers():
            yt = transformer.transform(yt)
        return yt

    def inverse_transform(self, y):
        self.check_is_fitted()
        yt = check_y(y)
        for step_idx, name, transformer in self._iter_transformers(
                reverse=True):
            yt = transformer.inverse_transform(yt)
        return yt

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('steps', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params('steps', **kwargs)
        return self
