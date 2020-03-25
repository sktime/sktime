#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["TransformedTargetForecaster"]

from itertools import islice
from sklearn.base import clone
from sktime.base import BaseComposition
from sktime.forecasting.base.base import BaseForecaster
from sktime.forecasting.base.sktime import BaseSktimeForecaster
from sktime.forecasting.base.meta import MetaForecasterMixin
from sktime.forecasting.base.sktime import OptionalForecastingHorizonMixin
from sktime.forecasting.base.base import DEFAULT_ALPHA
from sktime.transformers.detrend._base import BaseSeriesToSeriesTransformer
from sktime.utils.validation.forecasting import check_y


class TransformedTargetForecaster(MetaForecasterMixin, OptionalForecastingHorizonMixin, BaseSktimeForecaster,
                                  BaseComposition):
    """Meta-estimator for forecasting transformed time series."""

    _required_parameters = ("steps",)

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

        allowed_transformer_type = BaseSeriesToSeriesTransformer
        for t in transformers:
            # Transformers must be endog/exog transformers
            if not isinstance(t, allowed_transformer_type):
                raise TypeError(f"All intermediate steps should be "
                                f"instances of {allowed_transformer_type}, but "
                                f"transformer: {t} is not.")

        allowed_forecaster_type = BaseForecaster
        if not isinstance(forecaster, allowed_forecaster_type):
            raise TypeError(
                f"Last step of {self.__class__.__name__} must be of type: {allowed_forecaster_type}, "
                f"but forecaster: {forecaster} is not.")

        # Shallow copy
        return list(self.steps)

        # self.forecaster = forecaster
        # self.transformer = transformer
        # self.transformer_ = clone(self.transformer)
        # self.forecaster_ = clone(self.forecaster)

    def _iter(self):
        stop = len(self.steps_) - 1
        for idx, (name, transformer) in enumerate(islice(self.steps_, 0, stop)):
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
        self.steps_ = self._check_steps()
        self._set_oh(y_train)
        self._set_fh(fh)

        # transform
        yt = check_y(y_train)
        for step_idx, name, transformer in self._iter():
            cloned_transformer = clone(transformer)
            yt = cloned_transformer.fit_transform(yt)
            self.steps_[step_idx] = (name, cloned_transformer)

        # fit forecaster
        name, forecaster = self.steps[-1]
        cloned_forecaster = clone(forecaster)
        cloned_forecaster.fit(yt, fh=fh, X_train=X_train)
        self.steps_[-1] = (name, cloned_forecaster)

        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()

        forecaster = self.steps_[-1][1]
        y_pred = forecaster.predict(fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha)

        for step_idx, name, transformer in self._iter():
            y_pred = transformer.inverse_transform(y_pred)

        return y_pred

    def update(self, y_new, X_new=None, update_params=False):
        self.check_is_fitted()
        self._set_oh(y_new)

        for step_idx, name, transformer in self._iter():
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
        for step_idx, name, transformer in self._iter():
            yt = transformer.transform(yt)
        return yt

    def inverse_transform(self, y):
        self.check_is_fitted()
        yt = check_y(y)
        for step_idx, name, transformer in self._iter():
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

