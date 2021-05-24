#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["STLForecaster"]
__author__ = ["Taiwo Owoseni"]
from sklearn.base import clone
from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.forecasting import check_y
from sktime.utils.validation.series import check_series
from sktime.utils import _has_tag


class STLForecaster(
    _OptionalForecastingHorizonMixin,
    _SktimeForecaster,
    _HeterogenousMetaEstimator,
    _SeriesToSeriesTransformer,
):
    """
    STL - Seasonal and Trend decomposition using Loess.

    STL is a method for decomposing data into three components
    a) Seasonal Component
    b) Trend Component
    c) Residual

    Parameter
    ---------
    Forecaster: any forecaster
    """

    _required_parameters = ["steps"]
    _tags = {"univariate-only": True}

    def __init__(self, estimator, steps):
        self.estimator = estimator
        self.steps = steps
        self.steps_ = None
        self.estimator_ = None
        super(STLForecaster, self).__init__()

    def _check_steps(self):
        names, transformers = zip(*self.steps)

        # validate names
        self._check_names(names)

        # validate transformers
        # cextends to valid for
        # - deseasonal and detrend

        valid_transformer_type = _SeriesToSeriesTransformer
        for transformer in transformers:
            if not isinstance(transformer, valid_transformer_type):
                raise TypeError(
                    f"All intermediate steps should be "
                    f"instances of {valid_transformer_type}, "
                    f"but transformer: {transformer} is not."
                )

    def _check_estimator(self):
        estimator = self.estimator

        # validates estimator
        valid_forecaster_type = BaseForecaster
        if not isinstance(estimator, valid_forecaster_type):
            raise TypeError(
                f"first argument must be of type: "
                f"{valid_forecaster_type}, "
                f"but forecaster: {estimator} is not."
            )
        # Shallow copy
        return list(self.steps)

    def _iter_transformers(self, reverse=False):

        steps = self.steps_
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

    def fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self.steps_ = self._check_steps()
        self._set_y_X(y, X)
        self._set_fh(fh)

        # transform
        yt = check_y(y)
        for step_idx, name, transformer in self._iter_transformers():
            t = clone(transformer)
            yt = t.fit_transform(yt)
            self.steps_[step_idx] = (name, t)

        # fit forecaster
        name, forecaster = self.estimator
        f = clone(forecaster)
        f.fit(yt, X, fh)
        self.estimator_ = f

        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        forecaster = self.estimator_
        y_pred = forecaster.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)

        for _, _, transformer in self._iter_transformers(reverse=True):
            # skip sktime transformers where inverse transform
            # is not wanted ur meaningful (e.g. Imputer, HampelFilter)
            if not _has_tag(transformer, "skip-inverse-transform"):
                y_pred = transformer.inverse_transform(y_pred)
        return y_pred

    def update(self, y, X=None, update_params=True):
        """Update fitted parameters

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

        for step_idx, name, transformer in self._iter_transformers():
            if hasattr(transformer, "update"):
                transformer.update(y, update_params=update_params)
                self.steps_[step_idx] = (name, transformer)

        forecaster = self.estimator_
        forecaster.update(y, update_params=update_params)
        self.estimator_ = forecaster
        return self

    def transform(self, Z, X=None):
        self.check_is_fitted()
        zt = check_series(Z, enforce_univariate=True)
        for _, _, transformer in self._iter_transformers():
            zt = transformer.transform(zt, X)
        return zt

    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        zt = check_series(Z, enforce_univariate=True)
        for _, _, transformer in self._iter_transformers(reverse=True):
            zt = transformer.inverse_transform(zt, X)
        return zt

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
        return self._get_params("steps", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params("steps", **kwargs)
        return self


# stlforecaster(forecaste, steps)
