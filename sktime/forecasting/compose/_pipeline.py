#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements pipelines for forecasting."""

__author__ = ["mloning", "aiwalter"]
__all__ = ["TransformedTargetForecaster", "ForecastingPipeline"]

from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster
from sktime.transformations.base import BaseTransformer, _SeriesToSeriesTransformer
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

        valid_transformer_type = BaseTransformer
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

    # both children use the same step params for testing, so putting it here
    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.preprocessing import StandardScaler

        from sktime.forecasting.arima import ARIMA
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor
        from sktime.transformations.series.exponent import ExponentTransformer

        # StandardScaler does not skip fit, NaiveForecaster is not probabilistic
        STEPS1 = [
            ("transformer", TabularToSeriesAdaptor(StandardScaler())),
            ("forecaster", NaiveForecaster()),
        ]
        params1 = {"steps": STEPS1}

        # ARIMA has probabilistic methods, ExponentTransformer skips fit
        STEPS2 = [
            ("transformer", ExponentTransformer()),
            ("forecaster", ARIMA()),
        ]
        params2 = {"steps": STEPS2}

        return [params1, params2]


# we ensure that internally we convert to pd.DataFrame for now
SUPPORTED_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


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
        "y_inner_mtype": SUPPORTED_MTYPES,
        "X_inner_mtype": SUPPORTED_MTYPES,
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
    }

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_steps()
        super(ForecastingPipeline, self).__init__()
        _, forecaster = self.steps[-1]
        tags_to_clone = [
            "scitype:y",  # which y are fine? univariate/multivariate/both
            "ignores-exogeneous-X",  # does estimator ignore the exogeneous X?
            "capability:pred_int",  # can the estimator produce prediction intervals?
            "handles-missing-data",  # can estimator handle missing data?
            "requires-fh-in-fit",  # is forecasting horizon already required in fit?
            "X-y-must-have-same-index",  # can estimator handle different X/y index?
            "enforce_index_type",  # index type that needs to be enforced in X/y
        ]
        self.clone_tags(forecaster, tags_to_clone)
        self._anytagis_then_set("fit_is_empty", False, True, steps)

    @property
    def forecaster_(self):
        """Return reference to the forecaster in the pipeline. Valid after _fit."""
        return self.steps_[-1][1]

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
        # If X is not given or ignored, just passthrough the data without transformation
        if self._X is not None and not self.get_tag("ignores-exogeneous-X"):
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

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, required
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        X = self._transform(X=X)
        return self.forecaster_.predict(fh, X)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        X = self._transform(X=X)
        return self.forecaster_.predict_quantiles(fh=fh, X=X, alpha=alpha)

    def _predict_interval(self, fh, X=None, coverage=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        X = self._transform(X=X)
        return self.forecaster_.predict_interval(fh=fh, X=X, coverage=coverage)

    def _predict_var(self, fh, X=None, cov=False):
        """Forecast variance at future horizon.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on `cov` variable
            If cov=False:
                Column names are exactly those of `y` passed in `fit`/`update`.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh. Entries are variance forecasts, for var in col index.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
        """
        X = self._transform(X=X)
        return self.forecaster_.predict_var(fh=fh, X=X, cov=cov)

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : tfp Distribution object
            if marginal=True:
                batch shape is 1D and same length as fh
                event shape is 1D, with length equal number of variables being forecast
                i-th (batch) distribution is forecast for i-th entry of fh
                j-th (event) index is j-th variable, order as y in `fit`/`update`
            if marginal=False:
                there is a single batch
                event shape is 2D, of shape (len(fh), no. variables)
                i-th (event dim 1) distribution is forecast for i-th entry of fh
                j-th (event dim 1) index is j-th variable, order as y in `fit`/`update`
        """
        X = self._transform(X=X)
        return self.forecaster_.predict_proba(fh=fh, X=X, marginal=marginal)

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
        if X is not None:
            for _, _, transformer in self._iter_transformers():
                if hasattr(transformer, "update"):
                    transformer.update(X=X, y=y, update_params=update_params)
                    X = transformer.transform(X=X, y=y)

        _, forecaster = self.steps_[-1]
        forecaster.update(y=y, X=X, update_params=update_params)
        return self

    def _transform(self, X=None, y=None):
        # If X is not given or ignored, just passthrough the data without transformation
        if self._X is not None and not self.get_tag("ignores-exogeneous-X"):
            for _, _, transformer in self._iter_transformers():
                X = transformer.transform(X=X, y=y)
        return X


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
        "y_inner_mtype": SUPPORTED_MTYPES,
        "X_inner_mtype": SUPPORTED_MTYPES,
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
    }

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_steps()
        super(TransformedTargetForecaster, self).__init__()
        _, forecaster = self.steps[-1]
        tags_to_clone = [
            "scitype:y",  # which y are fine? univariate/multivariate/both
            "ignores-exogeneous-X",  # does estimator ignore the exogeneous X?
            "capability:pred_int",  # can the estimator produce prediction intervals?
            "handles-missing-data",  # can estimator handle missing data?
            "requires-fh-in-fit",  # is forecasting horizon already required in fit?
            "X-y-must-have-same-index",  # can estimator handle different X/y index?
            "enforce_index_type",  # index type that needs to be enforced in X/y
        ]
        self.clone_tags(forecaster, tags_to_clone)
        self._anytagis_then_set("fit_is_empty", False, True, steps)

    @property
    def forecaster_(self):
        """Return reference to the forecaster in the pipeline. Valid after _fit."""
        return self.steps_[-1][1]

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
            y = t.fit_transform(X=y, y=X)
            self.steps_[step_idx] = (name, t)

        # fit forecaster
        name, forecaster = self.steps[-1]
        f = clone(forecaster)
        f.fit(y=y, X=X, fh=fh)
        self.steps_[-1] = (name, f)
        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_pred = self.forecaster_.predict(fh=fh, X=X)
        # inverse transform y_pred
        y_pred = self._get_inverse_transform(y_pred, X)
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
        for _, _, transformer in self._iter_transformers():
            if hasattr(transformer, "update"):
                transformer.update(X=y, y=X, update_params=update_params)
                y = transformer.transform(X=y, y=X)

        self.forecaster_.update(y=y, X=X, update_params=update_params)
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

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        pred_int = self.forecaster_.predict_quantiles(fh=fh, X=X, alpha=alpha)
        pred_int_transformed = self._get_inverse_transform(pred_int)
        return pred_int_transformed

    def _predict_interval(self, fh, X=None, coverage=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        pred_int = self.forecaster_.predict_interval(fh=fh, X=X, coverage=coverage)
        pred_int_transformed = self._get_inverse_transform(pred_int)
        return pred_int_transformed
