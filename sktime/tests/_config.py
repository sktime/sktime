#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["ESTIMATOR_TEST_PARAMS", "EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

import numpy as np
from hcrystalball.wrappers import HoltSmoothingWrapper
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.fbprophet import Prophet
from sktime.base import BaseEstimator
from sktime.classification.base import BaseClassifier
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.interval_based import RandomIntervalSpectralForest
from sktime.classification.interval_based._cif import CanonicalIntervalForest
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.classification.shapelet_based import ROCKETClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.bats import BATS
from sktime.forecasting.compose import DirectRegressionForecaster
from sktime.forecasting.compose import DirectTimeSeriesRegressionForecaster
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose import MultioutputRegressionForecaster
from sktime.forecasting.compose import RecursiveRegressionForecaster
from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose import StackingForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.hcrystalball import HCrystalBallForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.forecasting.model_selection import SingleWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.online_learning import OnlineEnsembleForecaster
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import sMAPE
from sktime.regression.base import BaseRegressor
from sktime.regression.compose import TimeSeriesForestRegressor
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformations.base import BaseTransformer
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.base import _PanelToTabularTransformer
from sktime.transformations.base import _SeriesToPrimitivesTransformer
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.transformations.panel.compose import ColumnTransformer
from sktime.transformations.panel.compose import (
    SeriesToPrimitivesRowTransformer,
)
from sktime.transformations.panel.compose import SeriesToSeriesRowTransformer
from sktime.transformations.panel.dictionary_based import SFA
from sktime.transformations.panel.interpolate import TSInterpolator
from sktime.transformations.panel.reduce import Tabularizer
from sktime.transformations.panel.shapelets import ContractedShapeletTransform
from sktime.transformations.panel.shapelets import ShapeletTransform
from sktime.transformations.panel.summarize import FittedParamExtractor
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.transformations.panel.tsfresh import (
    TSFreshRelevantFeatureExtractor,
)
from sktime.transformations.series.acf import AutoCorrelationTransformer
from sktime.transformations.series.acf import PartialAutoCorrelationTransformer
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.impute import Imputer


# The following estimators currently do not pass all unit tests
# What do they fail? ShapeDTW fails on 3d_numpy_input test, not set up for that
EXCLUDE_ESTIMATORS = [
    "ShapeDTW",
    "HIVECOTEV1",
    "ElasticEnsemble",
    "KNeighborsTimeSeriesClassifier",
    "ProximityForest",
    "ProximityStump",
    "ProximityTree",
]

EXCLUDED_TESTS = {
    "ShapeletTransformClassifier": ["check_fit_idempotent"],
    "ContractedShapeletTransform": ["check_fit_idempotent"],
}

# We here configure estimators for basic unit testing, including setting of
# required hyper-parameters and setting of hyper-parameters for faster training.
SERIES_TO_SERIES_TRANSFORMER = StandardScaler()
SERIES_TO_PRIMITIVES_TRANSFORMER = FunctionTransformer(
    np.mean, kw_args={"axis": 0}, check_inverse=False
)
TRANSFORMERS = [
    (
        "transformer1",
        SeriesToSeriesRowTransformer(
            SERIES_TO_SERIES_TRANSFORMER, check_transformer=False
        ),
    ),
    (
        "transformer2",
        SeriesToSeriesRowTransformer(
            SERIES_TO_SERIES_TRANSFORMER, check_transformer=False
        ),
    ),
]
REGRESSOR = LinearRegression()
TIME_SERIES_CLASSIFIER = TimeSeriesForest(n_estimators=3)
TIME_SERIES_CLASSIFIERS = [
    ("tsf1", TIME_SERIES_CLASSIFIER),
    ("tsf2", TIME_SERIES_CLASSIFIER),
]
FORECASTER = ExponentialSmoothing()
FORECASTERS = [("ses1", FORECASTER), ("ses2", FORECASTER)]
STEPS = [
    ("transformer", Detrender(ThetaForecaster())),
    ("forecaster", NaiveForecaster()),
]
ESTIMATOR_TEST_PARAMS = {
    OnlineEnsembleForecaster: {"forecasters": FORECASTERS},
    FeatureUnion: {"transformer_list": TRANSFORMERS},
    DirectRegressionForecaster: {"regressor": REGRESSOR},
    MultioutputRegressionForecaster: {"regressor": REGRESSOR},
    RecursiveRegressionForecaster: {"regressor": REGRESSOR},
    DirectTimeSeriesRegressionForecaster: {
        "regressor": make_pipeline(Tabularizer(), REGRESSOR)
    },
    RecursiveTimeSeriesRegressionForecaster: {
        "regressor": make_pipeline(Tabularizer(), REGRESSOR)
    },
    TransformedTargetForecaster: {"steps": STEPS},
    EnsembleForecaster: {"forecasters": FORECASTERS},
    StackingForecaster: {"forecasters": FORECASTERS, "final_regressor": REGRESSOR},
    Detrender: {"forecaster": FORECASTER},
    ForecastingGridSearchCV: {
        "forecaster": NaiveForecaster(strategy="mean"),
        "cv": SingleWindowSplitter(fh=1),
        "param_grid": {"window_length": [2, 5]},
        "scoring": sMAPE(),
    },
    ForecastingRandomizedSearchCV: {
        "forecaster": NaiveForecaster(strategy="mean"),
        "cv": SingleWindowSplitter(fh=1),
        "param_distributions": {"window_length": [2, 5]},
        "scoring": sMAPE(),
    },
    TabularToSeriesAdaptor: {"transformer": StandardScaler()},
    ColumnEnsembleClassifier: {
        "estimators": [
            (name, estimator, 0) for (name, estimator) in TIME_SERIES_CLASSIFIERS
        ]
    },
    FittedParamExtractor: {
        "forecaster": FORECASTER,
        "param_names": ["initial_level"],
    },
    SeriesToPrimitivesRowTransformer: {
        "transformer": SERIES_TO_PRIMITIVES_TRANSFORMER,
        "check_transformer": False,
    },
    SeriesToSeriesRowTransformer: {
        "transformer": SERIES_TO_SERIES_TRANSFORMER,
        "check_transformer": False,
    },
    ColumnTransformer: {
        "transformers": [(name, estimator, [0]) for name, estimator in TRANSFORMERS]
    },
    AutoARIMA: {
        "d": 0,
        "suppress_warnings": True,
        "max_p": 2,
        "max_q": 2,
        "seasonal": False,
    },
    ShapeletTransformClassifier: {"n_estimators": 3, "time_contract_in_mins": 0.125},
    ContractedShapeletTransform: {"time_contract_in_mins": 0.125},
    ShapeletTransform: {
        "max_shapelets_to_store_per_class": 1,
        "min_shapelet_length": 3,
        "max_shapelet_length": 4,
    },
    ROCKETClassifier: {"num_kernels": 100},
    TSFreshFeatureExtractor: {"disable_progressbar": True, "show_warnings": False},
    TSFreshRelevantFeatureExtractor: {
        "disable_progressbar": True,
        "show_warnings": False,
        "fdr_level": 0.01,
    },
    TSInterpolator: {"length": 10},
    RandomIntervalSpectralForest: {"n_estimators": 3, "acf_lag": 10, "min_interval": 5},
    SFA: {"return_pandas_data_series": True},
    ContractableBOSS: {"n_parameter_samples": 25, "max_ensemble_size": 5},
    TemporalDictionaryEnsemble: {
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 20,
    },
    TimeSeriesForest: {"n_estimators": 3},
    TimeSeriesForestClassifier: {"n_estimators": 3},
    TimeSeriesForestRegressor: {"n_estimators": 3},
    SupervisedTimeSeriesForest: {"n_estimators": 3},
    CanonicalIntervalForest: {"n_estimators": 3},
    DrCIF: {"n_estimators": 3},
    HCrystalBallForecaster: {"model": HoltSmoothingWrapper()},
    BATS: {
        "use_box_cox": False,
        "use_trend": False,
        "use_damped_trend": False,
        "sp": [],
        "use_arma_errors": False,
        "n_jobs": 1,
    },
    TBATS: {
        "use_box_cox": False,
        "use_trend": False,
        "use_damped_trend": False,
        "sp": [],
        "use_arma_errors": False,
        "n_jobs": 1,
    },
    Prophet: {
        "n_changepoints": 0,
        "yearly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "uncertainty_samples": 1000,
        "verbose": False,
    },
    PartialAutoCorrelationTransformer: {"n_lags": 1},
    AutoCorrelationTransformer: {"n_lags": 1},
    Imputer: {"method": "mean"},
}

# These methods should not change the state of the estimator, that is, they should
# not change fitted parameters or hyper-parameters. They are also the methods that
# "apply" the fitted estimator to data and useful for checking results.
NON_STATE_CHANGING_METHODS = (
    "predict",
    "predict_proba",
    "decision_function",
    "transform",
    "inverse_transform",
)

# We use estimator tags in addition to class hierarchies to further distinguish
# estimators into different categories. This is useful for defining and running
# common tests for estimators with the same tags.
VALID_ESTIMATOR_TAGS = (
    "fit-in-transform",  # fitted in transform or non-fittable
    "univariate-only",
    "transform-returns-same-time-index",
)

# The following gives a list of valid estimator base classes.
VALID_TRANSFORMER_TYPES = (
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
    _PanelToTabularTransformer,
    _PanelToPanelTransformer,
)
VALID_ESTIMATOR_BASE_TYPES = (
    BaseClassifier,
    BaseRegressor,
    BaseForecaster,
    BaseTransformer,
)
VALID_ESTIMATOR_TYPES = (
    BaseEstimator,
    *VALID_ESTIMATOR_BASE_TYPES,
    *VALID_TRANSFORMER_TYPES,
)

VALID_ESTIMATOR_BASE_TYPE_LOOKUP = {
    "classifier": BaseClassifier,
    "regressor": BaseRegressor,
    "forecaster": BaseForecaster,
    "transformer": BaseTransformer,
}
