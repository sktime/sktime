# -*- coding: utf-8 -*-
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series

from sklearn.base import clone
from sklearn.utils.metaestimators import if_delegate_has_method


class OptionalPassthrough(_SeriesToSeriesTransformer):
    """A transformer to tune the implicite hyperparameter whether to use a
    particular transformer inside a pipeline (e.g. TranformedTargetForecaster)
    or not.This is achived by having the additional hyperparameter
    \"passthrough\" which can be added to a grid then (see example).

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series
    passthrough : bool
        This arg decides whether to apply the given transformer or to just
        passthrough the data (identity transformation)

    Examples
    ----------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.series.compose import OptionalPassthrough
    >>> from sktime.transformations.series.boxcox import BoxCoxTransformer
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.model_selection import (
        ForecastingGridSearchCV,
        SlidingWindowSplitter)

    >>> # create pipeline
    >>> pipe = TransformedTargetForecaster(steps=[
        ("boxcox", OptionalPassthrough(BoxCoxTransformer())),
        ("deseasonalizer", OptionalPassthrough(Deseasonalizer())),
        ("forecaster", NaiveForecaster())
        ]
    )

    >>> # putting it all together in a grid search
    >>> cv = SlidingWindowSplitter(
        initial_window=60, window_length=24,
        start_with_window=True,
        step_length=24)
    >>> param_grid = {
        "boxcox__transformer__method": ["pearsonr", "mle"],
        "boxcox__passthrough" : [True, False],
        "deseasonalizer__passthrough" : [True, False],
        'forecaster__strategy': ["drift", "mean", "last"],
    }
    >>> gscv = ForecastingGridSearchCV(
        forecaster=pipe,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1)
    """

    _required_parameters = ["transformer"]
    _tags = {
        "univariate-only": True,
        "fit-in-transform": True,
    }

    def __init__(self, transformer, passthrough=True):
        self.transformer = transformer
        self.transformer_ = None
        self.passthrough = passthrough
        self._is_fitted = False
        super(OptionalPassthrough, self).__init__()

    def fit(self, Z, X=None):
        self.transformer_ = clone(self.transformer)
        if not self.passthrough:
            self.transformer_.fit(Z, X)
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        if not self.passthrough:
            z = self.transformer_.transform(z, X)
        return z

    @if_delegate_has_method(delegate="transformer")
    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        if not self.passthrough:
            z = self.transformer_.inverse_transform(Z, X=None)
        return z
