#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["ForecastingPipeline"]

from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.forecasting import check_y, check_y_X
from sktime.utils.validation.series import check_series
from sktime.utils import _has_tag


class ForecastingPipeline(
    _OptionalForecastingHorizonMixin,
    _SktimeForecaster,
    _HeterogenousMetaEstimator,
    _SeriesToSeriesTransformer,
):
    _required_parameters = ["steps"]

    def __init__(self, steps):
        self.steps = steps
        self._steps = None
        super(ForecastingPipeline, self).__init__()

    def _check_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._check_names(names)

        # validate estimators
        transformers = estimators[:-1]
        forecaster = estimators[-1]

        valid_transformer_type = _SeriesToSeriesTransformer
        for transformer in transformers:
            if not isinstance(transformer, valid_transformer_type):
                raise TypeError(
                    f"All intermediate steps should be "
                    f"instances of {valid_transformer_type}, "
                    f"but transformer: {transformer} is not."
                )

        valid_forecaster_type = BaseForecaster
        if not isinstance(forecaster, valid_forecaster_type):
            raise TypeError(
                f"Last step of {self.__class__.__name__} must be of type: "
                f"{valid_forecaster_type}, "
                f"but forecaster: {forecaster} is not."
            )

        # Shallow copy
        return list(self.steps)

    def _iter_transformers(self, reverse=False):

        # exclude final forecaster
        steps = self._steps[:-1]

        if reverse:
            steps = reversed(steps)

        for idx, (name, transformer_y, transformer_X, identifier) in enumerate(steps):
            yield idx, name, transformer_y, transformer_X, identifier

    def _set_steps(self):
        # Adds an additional transformer so that we have then transformer_y
        # and transformer_X to iterate over
        _steps = []
        # exclude final forecaster
        for step in self._steps[:-1]:
            name, transformer, identifier = step
            step = (name, transformer, transformer, identifier)
            _steps.append(step)
        _steps.append(self._steps[-1])
        self._steps = _steps

    def __len__(self):
        """
        Returns the length of the Pipeline
        """
        return len(self.steps)

    @property
    def named_steps(self):
        """Map the steps to a dictionary"""
        return dict(self.steps)

    @property
    def steps_(self):
        """Removes duplicated transformer due to having
        transformer_y and transformer_X

        Returns
        -------
        list of tuples
        """
        steps_ = []
        for step in self._steps[:-1]:
            name, transformer, _, identifier = step
            step = (name, transformer, identifier)
            steps_.append(step)
        steps_.append(self._steps[-1])
        return steps_

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
        self._steps = self._check_steps()
        self._set_steps()
        self._set_y_X(y, X)
        self._set_fh(fh)
        # transform
        yt = check_y(y)
        if X is not None:
            yt, X = check_y_X(y, X)
        for (
            step_idx,
            name,
            transformer_y,
            transformer_X,
            identifier,
        ) in self._iter_transformers():
            transformer_y = clone(transformer_y)
            transformer_X = clone(transformer_X)
            if "y" in identifier:
                yt = transformer_y.fit_transform(yt, X)
            if "X" in identifier:
                X = transformer_X.fit_transform(X)
            self._steps[step_idx] = (name, transformer_y, transformer_X, identifier)

        # fit forecaster
        name, forecaster = self.steps[-1]
        f = clone(forecaster)
        f.fit(yt, X, fh)
        self._steps[-1] = (name, f)

        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        forecaster = self._steps[-1][1]
        y_pred = forecaster.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)

        for _, _, transformer_y, transformer_X, identifier in self._iter_transformers(
            reverse=True
        ):
            # skip sktime transformers where inverse transform
            # is not wanted ur meaningful (e.g. Imputer, HampelFilter)
            if "y" in identifier and not _has_tag(
                transformer_y, "skip-inverse-transform"
            ):
                y_pred = transformer_y.inverse_transform(y_pred, X)
            if "X" in identifier and not _has_tag(
                transformer_X, "skip-inverse-transform"
            ):
                X = transformer_X.inverse_transform(X)
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

        for (
            step_idx,
            name,
            transformer_y,
            transformer_X,
            identifier,
        ) in self._iter_transformers():
            if hasattr(transformer_y, "update"):
                transformer_y.update(y, update_params=update_params)
                transformer_X.update(X, update_params=update_params)
                self._steps[step_idx] = (name, transformer_y, transformer_X, identifier)

        name, forecaster = self._steps[-1]
        forecaster.update(y, update_params=update_params)
        self._steps[-1] = (name, forecaster)
        return self

    def transform(self, Z, X=None):
        self.check_is_fitted()
        zt = check_series(Z, enforce_univariate=True)
        for _, _, transformer_y, transformer_X, identifier in self._iter_transformers():
            if "y" in identifier:
                zt = transformer_y.transform(zt, X)
            if "X" in identifier:
                X = transformer_X.transform(X)
        return zt

    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        zt = check_series(Z, enforce_univariate=True)
        for _, _, transformer_y, transformer_X, identifier in self._iter_transformers(
            reverse=True
        ):
            if "y" in identifier:
                zt = transformer_y.inverse_transform(zt, X)
            if "X" in identifier:
                X = transformer_X.inverse_transform(X)
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
