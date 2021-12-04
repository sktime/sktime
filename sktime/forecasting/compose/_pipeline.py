#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements pipelines for forecasting."""

__author__ = ["mloning", "aiwalter"]
__all__ = ["TransformedTargetForecaster", "ForecastingPipeline"]

from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import DEFAULT_ALPHA, BaseForecaster
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class _Pipeline(
    BaseForecaster,
    _HeterogenousMetaEstimator,
):
    def _check_steps(self):
        """Check Steps.

        Parameters
        ----------
        self : an instance of self

        Returns
        -------
        step : Returns step.
        """
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
        steps = self.steps_[:-1]

        if reverse:
            steps = reversed(steps)

        for idx, (name, transformer) in enumerate(steps):
            yield idx, name, transformer

    def __len__(self):
        """Return the length of the Pipeline."""
        return len(self.steps)

    def _get_inverse_transform(self, y, X=None):
        """Iterate over transformers.

        Inverse transform y (used for y_pred and pred_int)

        Parameters
        ----------
        y : pd.Series, pd.DataFrame
            Target series
        X : pd.Series, pd.DataFrame
            Exogenous series.

        Returns
        -------
        y : pd.Series, pd.DataFrame
            Inverse transformed y
        """
        for _, _, transformer in self._iter_transformers(reverse=True):
            # skip sktime transformers where inverse transform
            # is not wanted ur meaningful (e.g. Imputer, HampelFilter)
            skip_trafo = transformer.get_tag("skip-inverse-transform", False)
            if not skip_trafo:
                y = transformer.inverse_transform(y, X)
        return y

    @property
    def named_steps(self):
        """Map the steps to a dictionary."""
        return dict(self.steps)

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


class ForecastingPipeline(_Pipeline):
    """Pipeline for forecasting with exogenous data.

    ForecastingPipeline is only applying the given transformers
    to X. The forecaster can also be a TransformedTargetForecaster containing
    transformers to transform y.

    Parameters
    ----------
    steps : list
        List of tuples like ("name", forecaster/transformer)

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> y, X = load_longley()
    >>> y_train, _, X_train, X_test = temporal_train_test_split(y, X)
    >>> fh = ForecastingHorizon(X_test.index, is_relative=False)
    >>> pipe = ForecastingPipeline(steps=[
    ...     ("imputer", Imputer(method="mean")),
    ...     ("minmaxscaler", TabularToSeriesAdaptor(MinMaxScaler())),
    ...     ("forecaster", NaiveForecaster(strategy="drift"))])
    >>> pipe.fit(y_train, X_train)
    ForecastingPipeline(...)
    >>> y_pred = pipe.predict(fh=fh, X=X_test)
    """

    _required_parameters = ["steps"]
    _tags = {
        "scitype:y": "both",
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_steps()
        super(ForecastingPipeline, self).__init__()
        _, forecaster = self.steps[-1]
        self.clone_tags(forecaster)

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, required
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        # If X is not given, just passthrough the data without transformation
        if self._X is not None:
            # transform X
            for step_idx, name, transformer in self._iter_transformers():
                t = clone(transformer)
                X = t.fit_transform(X=X, y=y)
                self.steps_[step_idx] = (name, t)

        # fit forecaster
        name, forecaster = self.steps[-1]
        f = clone(forecaster)
        f.fit(y, X, fh)
        self.steps_[-1] = (name, f)

        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, required
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=DEFAULT_ALPHA)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        forecaster = self.steps_[-1][1]

        # If X is not given, just passthrough the data without transformation
        if self._X is not None:
            # transform X before doing prediction
            for _, _, transformer in self._iter_transformers():
                X = transformer.transform(Z=X)

        return forecaster.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame, required
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        # If X is not given, just passthrough the data without transformation
        if self._X is not None:
            for step_idx, name, transformer in self._iter_transformers():
                if hasattr(transformer, "update"):
                    transformer.update(X, update_params=update_params)
                    self.steps_[step_idx] = (name, transformer)

        name, forecaster = self.steps_[-1]
        forecaster.update(y=y, X=X, update_params=update_params)
        self.steps_[-1] = (name, forecaster)
        return self


# removed transform and inverse_transform as long as y can only be a pd.Series
# def transform(self, Z, X=None):
#     self.check_is_fitted()
#     Zt = check_series(Z, enforce_multivariate=True)
#     for _, _, transformer in self._iter_transformers():
#         Zt = transformer.transform(Zt)
#     return Zt

# def inverse_transform(self, Z, X=None):
#     self.check_is_fitted()
#     Zt = check_series(Z, enforce_multivariate=True)
#     for _, _, transformer in self._iter_transformers(reverse=True):
#         if not _has_tag(transformer, "skip-inverse-transform"):
#             Zt = transformer.inverse_transform(Zt)
#     return Zt


class TransformedTargetForecaster(_Pipeline, _SeriesToSeriesTransformer):
    """Meta-estimator for forecasting transformed time series.

    Pipeline functionality to apply transformers to the target series. The
    X data is not transformed. If you want to transform X, please use the
    ForecastingPipeline.

    Parameters
    ----------
    steps : list
        List of tuples like ("name", forecaster/transformer)

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> y = load_airline()
    >>> pipe = TransformedTargetForecaster(steps=[
    ...     ("imputer", Imputer(method="mean")),
    ...     ("detrender", Deseasonalizer()),
    ...     ("forecaster", NaiveForecaster(strategy="drift"))])
    >>> pipe.fit(y)
    TransformedTargetForecaster(...)
    >>> y_pred = pipe.predict(fh=[1,2,3])
    """

    _required_parameters = ["steps"]
    _tags = {
        "scitype:y": "both",
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "ignores-exogeneous-X": True,
        "univariate-only": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_steps()
        super(TransformedTargetForecaster, self).__init__()
        _, forecaster = self.steps[-1]
        self.clone_tags(forecaster)

    def _fit(self, y, X=None, fh=None):
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
        # transform
        for step_idx, name, transformer in self._iter_transformers():
            t = clone(transformer)
            y = t.fit_transform(y, X)
            self.steps_[step_idx] = (name, t)

        # fit forecaster
        name, forecaster = self.steps[-1]
        f = clone(forecaster)
        f.fit(y, X, fh)
        self.steps_[-1] = (name, f)
        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=DEFAULT_ALPHA)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        forecaster = self.steps_[-1][1]
        if return_pred_int:
            y_pred, pred_int = forecaster.predict(
                fh, X, return_pred_int=return_pred_int, alpha=alpha
            )
            # inverse transform pred_int
            pred_int["lower"] = self._get_inverse_transform(pred_int["lower"], X)
            pred_int["upper"] = self._get_inverse_transform(pred_int["upper"], X)
        else:
            y_pred = forecaster.predict(
                fh, X, return_pred_int=return_pred_int, alpha=alpha
            )
        # inverse transform y_pred
        y_pred = self._get_inverse_transform(y_pred, X)
        if return_pred_int:
            return y_pred, pred_int
        else:
            return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        for step_idx, name, transformer in self._iter_transformers():
            if hasattr(transformer, "update"):
                transformer.update(y, X, update_params=update_params)
                self.steps_[step_idx] = (name, transformer)

        name, forecaster = self.steps_[-1]
        forecaster.update(y=y, X=X, update_params=update_params)
        self.steps_[-1] = (name, forecaster)
        return self

    def transform(self, Z, X=None):
        """Return transformed version of input series `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the transformation on.
        X : pd.DataFrame, default=None
            Exogenous data used in transformation.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            Transformed version of input series `Z`.
        """
        self.check_is_fitted()
        zt = check_series(Z)
        for _, _, transformer in self._iter_transformers():
            zt = transformer.transform(zt, X)
        return zt

    def inverse_transform(self, Z, X=None):
        """Reverse transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to reverse the transformation on.
        X : pd.DataFrame, default=None
            Exogenous data used in transformation.

        Returns
        -------
        Z_inv : pd.Series or pd.DataFrame
            The reconstructed timeseries after the transformation has been reversed.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        return self._get_inverse_transform(Z, X)
