#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "ESTIMATOR_TEST_PARAMS",
    "EXCLUDED"
]

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import DirectRegressionForecaster
from sktime.forecasting.compose import DirectTimeSeriesRegressionForecaster
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose import RecursiveRegressionForecaster
from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose import StackingForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.model_selection import SingleWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import sMAPE
from sktime.transformers.series_as_features.compose import ColumnTransformer
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.reduce import Tabularizer
from sktime.transformers.series_as_features.shapelets import \
    ContractedShapeletTransform
from sktime.transformers.series_as_features.shapelets import ShapeletTransform
from sktime.transformers.series_as_features.summarize import \
    FittedParamExtractor
from sktime.transformers.series_as_features.summarize import \
    TSFreshFeatureExtractor
from sktime.transformers.series_as_features.summarize import \
    TSFreshRelevantFeatureExtractor
from sktime.transformers.single_series.adapt import \
    SingleSeriesTransformAdaptor
from sktime.transformers.single_series.detrend import Detrender
from sktime.transformers.series_as_features.interpolate import TSInterpolator

# TODO fix estimators to pass all tests
EXCLUDED = [
    'ContractedShapeletTransform',
    'ElasticEnsemble',
    'KNeighborsTimeSeriesClassifier',
    # 'MrSEQLClassifier',
    'ProximityForest',
    'ProximityStump',
    'ProximityTree',
    'ShapeletTransform',
    'ShapeletTransformClassifier',
]

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
        {"time_contract_in_mins": 0.1},
    ContractedShapeletTransform:
        {"time_contract_in_mins": 0.1},
    ShapeletTransform:
        {"max_shapelets_to_store_per_class": 1},
    TSFreshFeatureExtractor:
        {"disable_progressbar": True, "show_warnings": False},
    TSFreshRelevantFeatureExtractor:
        {"disable_progressbar": True, "show_warnings": False,
         "ml_task": "classification", },
    TSInterpolator: {"length": 10},
    RandomIntervalSpectralForest: {
        "acf_lag": 10}
}
