#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "ESTIMATOR_TEST_PARAMS",
    "EXCLUDED_ESTIMATORS",
    "EXCLUDED_TESTS"
]

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import (
    DirectRegressionForecaster, DirectTimeSeriesRegressionForecaster,
    EnsembleForecaster, RecursiveRegressionForecaster,
    RecursiveTimeSeriesRegressionForecaster, StackingForecaster,
    TransformedTargetForecaster)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (ForecastingGridSearchCV,
                                                SingleWindowSplitter)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import sMAPE
from sktime.transformers.series_as_features.compose import (ColumnTransformer,
                                                            RowTransformer)
from sktime.transformers.series_as_features.interpolate import TSInterpolator
from sktime.transformers.series_as_features.reduce import Tabularizer
from sktime.transformers.series_as_features.dictionary_based import SFA
from sktime.transformers.series_as_features.shapelets import (
    ContractedShapeletTransform, ShapeletTransform)
from sktime.transformers.series_as_features.summarize import (
    FittedParamExtractor, TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor)
from sktime.transformers.single_series.adapt import \
    SingleSeriesTransformAdaptor
from sktime.transformers.single_series.detrend import Detrender

# TODO fix estimators to pass all tests
EXCLUDED_ESTIMATORS = [
    'ElasticEnsemble',
    'KNeighborsTimeSeriesClassifier',
    'ProximityForest',
    'ProximityStump',
    'ProximityTree',
]

EXCLUDED_TESTS = {
    "ShapeletTransformClassifier": ["check_fit_idempotent"],
    "ContractedShapeletTransform": ["check_fit_idempotent"],
}

TRANSFORMER = StandardScaler()
TRANSFORMERS = [
    ("t1", RowTransformer(TRANSFORMER)),
    ("t2", RowTransformer(TRANSFORMER)),
]
REGRESSOR = LinearRegression()
TIME_SERIES_CLASSIFIER = TimeSeriesForest(random_state=1)
TIME_SERIES_CLASSIFIERS = [
    ("tsf1", TIME_SERIES_CLASSIFIER),
    ("tsf2", TIME_SERIES_CLASSIFIER)
]
FORECASTER = ExponentialSmoothing()
FORECASTERS = [
    ("ses1", FORECASTER),
    ("ses2", FORECASTER)
]
STEPS = [
    ("t", Detrender(ThetaForecaster())),
    ("f", NaiveForecaster())
]
ESTIMATOR_TEST_PARAMS = {
    DirectRegressionForecaster:
        {"regressor": REGRESSOR},
    RecursiveRegressionForecaster:
        {"regressor": REGRESSOR},
    DirectTimeSeriesRegressionForecaster:
        {"regressor": make_pipeline(Tabularizer(), REGRESSOR)},
    RecursiveTimeSeriesRegressionForecaster:
        {"regressor": make_pipeline(Tabularizer(), REGRESSOR)},
    TransformedTargetForecaster:
        {"steps": STEPS},
    EnsembleForecaster:
        {"forecasters": FORECASTERS},
    StackingForecaster:
        {"forecasters": FORECASTERS, "final_regressor": REGRESSOR},
    Detrender:
        {"forecaster": FORECASTER},
    ForecastingGridSearchCV: {
        "forecaster": NaiveForecaster(strategy="mean"),
        "cv": SingleWindowSplitter(fh=1),
        "param_grid": {"window_length": [2, 5]},
        "scoring": sMAPE()},
    SingleSeriesTransformAdaptor:
        {"transformer": StandardScaler()},
    ColumnEnsembleClassifier:
        {"estimators": [(name, estimator, 0) for (name, estimator) in
                        TIME_SERIES_CLASSIFIERS]},
    FittedParamExtractor:
        {"forecaster": FORECASTER, "param_names": ["smoothing_level"]},
    RowTransformer:
        {"transformer": TRANSFORMER},
    ColumnTransformer:
        {"transformers": [(name, estimator, [0]) for name, estimator in
                          TRANSFORMERS]},
    # ARIMA requires d > start where start = 0 for full in-sample predictions
    AutoARIMA:
        {"d": 0, "suppress_warnings": True},
    ShapeletTransformClassifier:
        {"time_contract_in_mins": 0.125},
    ContractedShapeletTransform:
        {"time_contract_in_mins": 0.125},
    ShapeletTransform:
        {"max_shapelets_to_store_per_class": 1,
         "min_shapelet_length": 3,
         "max_shapelet_length": 4},
    TSFreshFeatureExtractor:
        {"disable_progressbar": True, "show_warnings": False},
    TSFreshRelevantFeatureExtractor:
        {"disable_progressbar": True, "show_warnings": False,
         "ml_task": "classification", },
    TSInterpolator: {"length": 10},
    RandomIntervalSpectralForest: {
        "acf_lag": 10},
    SFA: {
        "return_pandas_data_series": True}
}
