#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["ESTIMATOR_TEST_PARAMS", "EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

import numpy as np
from hcrystalball.wrappers import HoltSmoothingWrapper
from pyod.models.knn import KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sktime.annotation.adapters import PyODAnnotator
from sktime.annotation.clasp import ClaSPSegmentation
from sktime.base import BaseEstimator
from sktime.classification.compose import (
    ColumnEnsembleClassifier,
    ComposableTimeSeriesForestClassifier,
)
from sktime.classification.dictionary_based import (
    MUSE,
    WEASEL,
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
)
from sktime.classification.feature_based import (
    Catch22Classifier,
    MatrixProfileClassifier,
    SignatureClassifier,
    TSFreshClassifier,
)
from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sktime.classification.interval_based import (
    CanonicalIntervalForest,
    DrCIF,
    RandomIntervalSpectralEnsemble,
    RandomIntervalSpectralForest,
    SupervisedTimeSeriesForest,
)
from sktime.classification.interval_based import TimeSeriesForestClassifier as TSFC
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.dists_kernels.compose_tab_to_panel import AggrDist
from sktime.dists_kernels.scipy_dist import ScipyDist
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.bats import BATS
from sktime.forecasting.compose import (
    AutoEnsembleForecaster,
    ColumnEnsembleForecaster,
    DirectTabularRegressionForecaster,
    DirectTimeSeriesRegressionForecaster,
    DirRecTabularRegressionForecaster,
    DirRecTimeSeriesRegressionForecaster,
    EnsembleForecaster,
    ForecastingPipeline,
    MultioutputTabularRegressionForecaster,
    MultioutputTimeSeriesRegressionForecaster,
    MultiplexForecaster,
    RecursiveTabularRegressionForecaster,
    RecursiveTimeSeriesRegressionForecaster,
    StackingForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.hcrystalball import HCrystalBallForecaster
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ForecastingRandomizedSearchCV,
    SingleWindowSplitter,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.online_learning import OnlineEnsembleForecaster
from sktime.forecasting.structural import UnobservedComponents
from sktime.forecasting.tbats import TBATS
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.registry import (
    BASE_CLASS_LIST,
    BASE_CLASS_LOOKUP,
    ESTIMATOR_TAG_LIST,
    TRANSFORMER_MIXIN_LIST,
)
from sktime.regression.compose import ComposableTimeSeriesForestRegressor
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformations.panel.compose import (
    ColumnTransformer,
    SeriesToPrimitivesRowTransformer,
    SeriesToSeriesRowTransformer,
)
from sktime.transformations.panel.dictionary_based import SFA
from sktime.transformations.panel.interpolate import TSInterpolator
from sktime.transformations.panel.reduce import Tabularizer
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.transformations.panel.shapelets import (
    ContractedShapeletTransform,
    ShapeletTransform,
)
from sktime.transformations.panel.signature_based import SignatureTransformer
from sktime.transformations.panel.summarize import FittedParamExtractor
from sktime.transformations.panel.tsfresh import (
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor,
)
from sktime.transformations.series.acf import (
    AutoCorrelationTransformer,
    PartialAutoCorrelationTransformer,
)
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.clasp import ClaSPTransformer
from sktime.transformations.series.compose import (
    ColumnwiseTransformer,
    OptionalPassthrough,
)
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.feature_selection import FeatureSelection
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter

# The following estimators currently do not pass all unit tests
# What do they fail? ShapeDTW fails on 3d_numpy_input test, not set up for that
EXCLUDE_ESTIMATORS = [
    "ElasticEnsemble",
    "ProximityForest",
    "ProximityStump",
    "ProximityTree",
]


# This is temporary until BaseObject is implemented
DIST_KERNELS_IGNORE_TESTS = [
    "check_fit_updates_state",
    "_make_fit_args",
    "check_fit_returns_self",
    "check_raises_not_fitted_error",
    "check_fit_idempotent",
    "check_fit_does_not_overwrite_hyper_params",
    "check_methods_do_not_change_state",
    "check_persistence_via_pickle",
]


EXCLUDED_TESTS = {
    "ContractedShapeletTransform": ["check_fit_idempotent"],
    "ScipyDist": DIST_KERNELS_IGNORE_TESTS,
    "AggrDist": DIST_KERNELS_IGNORE_TESTS,
    "DistFromAligner": DIST_KERNELS_IGNORE_TESTS,
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
ANOMALY_DETECTOR = KNN()
TIME_SERIES_CLASSIFIER = TSFC(n_estimators=3)
TIME_SERIES_CLASSIFIERS = [
    ("tsf1", TIME_SERIES_CLASSIFIER),
    ("tsf2", TIME_SERIES_CLASSIFIER),
]
FORECASTER = NaiveForecaster()
FORECASTERS = [("f1", FORECASTER), ("f2", FORECASTER)]
STEPS = [
    ("transformer", TabularToSeriesAdaptor(StandardScaler())),
    ("forecaster", NaiveForecaster()),
]
ESTIMATOR_TEST_PARAMS = {
    ColumnEnsembleForecaster: {"forecasters": FORECASTER},
    OnlineEnsembleForecaster: {"forecasters": FORECASTERS},
    FeatureUnion: {"transformer_list": TRANSFORMERS},
    DirectTabularRegressionForecaster: {"estimator": REGRESSOR},
    MultioutputTabularRegressionForecaster: {"estimator": REGRESSOR},
    RecursiveTabularRegressionForecaster: {"estimator": REGRESSOR},
    DirRecTabularRegressionForecaster: {"estimator": REGRESSOR},
    DirectTimeSeriesRegressionForecaster: {
        "estimator": make_pipeline(Tabularizer(), REGRESSOR)
    },
    RecursiveTimeSeriesRegressionForecaster: {
        "estimator": make_pipeline(Tabularizer(), REGRESSOR)
    },
    MultioutputTimeSeriesRegressionForecaster: {
        "estimator": make_pipeline(Tabularizer(), REGRESSOR)
    },
    DirRecTimeSeriesRegressionForecaster: {
        "estimator": make_pipeline(Tabularizer(), REGRESSOR)
    },
    TransformedTargetForecaster: {"steps": STEPS},
    ForecastingPipeline: {"steps": STEPS},
    EnsembleForecaster: {"forecasters": FORECASTERS},
    StackingForecaster: {"forecasters": FORECASTERS},
    AutoEnsembleForecaster: {"forecasters": FORECASTERS},
    Detrender: {"forecaster": ExponentialSmoothing()},
    ForecastingGridSearchCV: {
        "forecaster": NaiveForecaster(strategy="mean"),
        "cv": SingleWindowSplitter(fh=1),
        "param_grid": {"window_length": [2, 5]},
        "scoring": MeanAbsolutePercentageError(symmetric=True),
    },
    ForecastingRandomizedSearchCV: {
        "forecaster": NaiveForecaster(strategy="mean"),
        "cv": SingleWindowSplitter(fh=1),
        "param_distributions": {"window_length": [2, 5]},
        "scoring": MeanAbsolutePercentageError(symmetric=True),
    },
    TabularToSeriesAdaptor: {"transformer": StandardScaler()},
    ColumnEnsembleClassifier: {
        "estimators": [
            (name, estimator, 0) for (name, estimator) in TIME_SERIES_CLASSIFIERS
        ]
    },
    FittedParamExtractor: {
        "forecaster": ExponentialSmoothing(),
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
    MultiplexForecaster: {
        "forecasters": [
            ("Naive_mean", NaiveForecaster(strategy="mean")),
            ("Naive_last", NaiveForecaster(strategy="last")),
            ("Naive_drift", NaiveForecaster(strategy="drift")),
        ],
        "selected_forecaster": "Naive_mean",
    },
    ShapeletTransformClassifier: {
        "estimator": RotationForest(n_estimators=3),
        "max_shapelets": 5,
        "n_shapelet_samples": 50,
        "batch_size": 20,
    },
    ContractedShapeletTransform: {"time_contract_in_mins": 0.025},
    ShapeletTransform: {
        "max_shapelets_to_store_per_class": 1,
        "min_shapelet_length": 3,
        "max_shapelet_length": 4,
    },
    RandomShapeletTransform: {
        "max_shapelets": 5,
        "n_shapelet_samples": 50,
        "batch_size": 20,
    },
    SignatureTransformer: {
        "augmentation_list": ("basepoint", "addtime"),
        "depth": 3,
        "window_name": "global",
    },
    SignatureClassifier: {
        "augmentation_list": ("basepoint", "addtime"),
        "depth": 3,
        "window_name": "global",
    },
    Catch22Classifier: {
        "estimator": RandomForestClassifier(n_estimators=3),
    },
    MatrixProfileClassifier: {
        "subsequence_length": 4,
    },
    TSFreshClassifier: {
        "estimator": RandomForestClassifier(n_estimators=3),
        "default_fc_parameters": "minimal",
    },
    RocketClassifier: {"num_kernels": 100},
    Arsenal: {"num_kernels": 50, "n_estimators": 3},
    HIVECOTEV1: {
        "stc_params": {
            "estimator": RotationForest(n_estimators=2),
            "max_shapelets": 5,
            "n_shapelet_samples": 20,
            "batch_size": 10,
        },
        "tsf_params": {"n_estimators": 2},
        "rise_params": {"n_estimators": 2},
        "cboss_params": {"n_parameter_samples": 4, "max_ensemble_size": 2},
    },
    HIVECOTEV2: {
        "stc_params": {
            "estimator": RotationForest(n_estimators=2),
            "max_shapelets": 5,
            "n_shapelet_samples": 20,
            "batch_size": 10,
        },
        "drcif_params": {"n_estimators": 2},
        "arsenal_params": {"num_kernels": 20, "n_estimators": 2},
        "tde_params": {
            "n_parameter_samples": 4,
            "max_ensemble_size": 2,
            "randomly_selected_params": 2,
        },
    },
    TSFreshFeatureExtractor: {"disable_progressbar": True, "show_warnings": False},
    TSFreshRelevantFeatureExtractor: {
        "disable_progressbar": True,
        "show_warnings": False,
        "fdr_level": 0.01,
    },
    TSInterpolator: {"length": 10},
    RandomIntervalSpectralForest: {"n_estimators": 3, "acf_lag": 10, "min_interval": 5},
    RandomIntervalSpectralEnsemble: {
        "n_estimators": 3,
        "acf_lag": 10,
        "min_interval": 5,
    },
    SFA: {"return_pandas_data_series": True},
    BOSSEnsemble: {"max_ensemble_size": 3},
    ContractableBOSS: {"n_parameter_samples": 10, "max_ensemble_size": 3},
    WEASEL: {"window_inc": 4},
    MUSE: {"window_inc": 4, "use_first_order_differences": False},
    TemporalDictionaryEnsemble: {
        "n_parameter_samples": 10,
        "max_ensemble_size": 3,
        "randomly_selected_params": 5,
    },
    TSFC: {"n_estimators": 3},
    ComposableTimeSeriesForestClassifier: {"n_estimators": 3},
    ComposableTimeSeriesForestRegressor: {"n_estimators": 3},
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
    UnobservedComponents: {"level": "local level"},
    PartialAutoCorrelationTransformer: {"n_lags": 1},
    AutoCorrelationTransformer: {"n_lags": 1},
    Imputer: {"method": "mean"},
    HampelFilter: {"window_length": 3},
    OptionalPassthrough: {"transformer": BoxCoxTransformer(), "passthrough": False},
    FeatureSelection: {"method": "all"},
    ColumnwiseTransformer: {"transformer": Detrender()},
    AggrDist: {"transformer": ScipyDist()},
    PyODAnnotator: {"estimator": ANOMALY_DETECTOR},
    ClaSPSegmentation: {"period_length": 5, "n_cps": 1},
    ClaSPTransformer: {"window_length": 5},
}

# We use estimator tags in addition to class hierarchies to further distinguish
# estimators into different categories. This is useful for defining and running
# common tests for estimators with the same tags.
VALID_ESTIMATOR_TAGS = tuple(ESTIMATOR_TAG_LIST)

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

# The following gives a list of valid estimator base classes.
VALID_TRANSFORMER_TYPES = tuple(TRANSFORMER_MIXIN_LIST)

VALID_ESTIMATOR_BASE_TYPES = tuple(BASE_CLASS_LIST)

VALID_ESTIMATOR_TYPES = (
    BaseEstimator,
    *VALID_ESTIMATOR_BASE_TYPES,
    *VALID_TRANSFORMER_TYPES,
)

VALID_ESTIMATOR_BASE_TYPE_LOOKUP = BASE_CLASS_LOOKUP
