# -*- coding: utf-8 -*-

__author__ = ["mloning"]
__all__ = ["ESTIMATOR_TEST_PARAMS", "EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

import numpy as np
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
from sktime.classification.distance_based import ElasticEnsemble
from sktime.classification.early_classification import (
    ProbabilityThresholdEarlyClassifier,
)
from sktime.classification.feature_based import (
    Catch22Classifier,
    MatrixProfileClassifier,
    RandomIntervalClassifier,
    SignatureClassifier,
    SummaryClassifier,
    TSFreshClassifier,
)
from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sktime.classification.interval_based import (
    CanonicalIntervalForest,
    DrCIF,
    RandomIntervalSpectralEnsemble,
    SupervisedTimeSeriesForest,
)
from sktime.classification.interval_based import TimeSeriesForestClassifier as TSFC
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.forecasting.compose import (
    DirectTabularRegressionForecaster,
    DirectTimeSeriesRegressionForecaster,
    DirRecTabularRegressionForecaster,
    DirRecTimeSeriesRegressionForecaster,
    MultioutputTabularRegressionForecaster,
    MultioutputTimeSeriesRegressionForecaster,
    RecursiveTabularRegressionForecaster,
    RecursiveTimeSeriesRegressionForecaster,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.structural import UnobservedComponents
from sktime.registry import (
    BASE_CLASS_LIST,
    BASE_CLASS_LOOKUP,
    ESTIMATOR_TAG_LIST,
    TRANSFORMER_MIXIN_LIST,
)
from sktime.regression.compose import ComposableTimeSeriesForestRegressor
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel.compose import (
    ColumnTransformer,
    SeriesToPrimitivesRowTransformer,
    SeriesToSeriesRowTransformer,
)
from sktime.transformations.panel.dictionary_based import SFA
from sktime.transformations.panel.interpolate import TSInterpolator
from sktime.transformations.panel.random_intervals import RandomIntervals
from sktime.transformations.panel.reduce import Tabularizer
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.transformations.panel.signature_based import SignatureTransformer
from sktime.transformations.panel.summarize import FittedParamExtractor
from sktime.transformations.panel.tsfresh import (
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor,
)
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.summarize import SummaryTransformer

# The following estimators currently do not pass all unit tests
# https://github.com/alan-turing-institute/sktime/issues/1627
EXCLUDE_ESTIMATORS = [
    "ProximityForest",
    "ProximityStump",
    "ProximityTree",
    # ConditionalDeseasonalizer and STLtransformer still need refactoring
    #  (see PR 1773, blocked through open discussion) escaping until then
    "ConditionalDeseasonalizer",
    "STLTransformer",
]


EXCLUDED_TESTS = {
    "FeatureUnion": ["test_fit_does_not_overwrite_hyper_params"],
    "StackingForecaster": ["test_predict_time_index_with_X"],
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
    ShapeletTransformClassifier: {
        "estimator": RotationForest(n_estimators=3),
        "max_shapelets": 5,
        "n_shapelet_samples": 50,
        "batch_size": 20,
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
    ElasticEnsemble: {
        "proportion_of_param_options": 0.01,
        "proportion_train_for_test": 0.1,
        "majority_vote": True,
        "distance_measures": ["dtw"],
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
    RandomIntervals: {
        "n_intervals": 3,
    },
    RandomIntervalClassifier: {
        "n_intervals": 3,
        "estimator": RandomForestClassifier(n_estimators=3),
        "interval_transformers": SummaryTransformer(
            summary_function=("mean", "min", "max"),
        ),
    },
    SummaryClassifier: {
        "estimator": RandomForestClassifier(n_estimators=3),
        "summary_functions": ("mean", "min", "max"),
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
    ProbabilityThresholdEarlyClassifier: {
        "classification_points": [3],
        "estimator": Catch22Classifier(
            estimator=RandomForestClassifier(n_estimators=2)
        ),
    },
    TSFreshFeatureExtractor: {"disable_progressbar": True, "show_warnings": False},
    TSFreshRelevantFeatureExtractor: {
        "disable_progressbar": True,
        "show_warnings": False,
        "fdr_level": 0.01,
    },
    TSInterpolator: {"length": 10},
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
    UnobservedComponents: {"level": "local level"},
    PyODAnnotator: {"estimator": ANOMALY_DETECTOR},
    ClaSPSegmentation: {"period_length": 5, "n_cps": 1},
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
VALID_TRANSFORMER_TYPES = tuple(TRANSFORMER_MIXIN_LIST) + (BaseTransformer,)

VALID_ESTIMATOR_BASE_TYPES = tuple(BASE_CLASS_LIST)

VALID_ESTIMATOR_TYPES = (
    BaseEstimator,
    *VALID_ESTIMATOR_BASE_TYPES,
    *VALID_TRANSFORMER_TYPES,
)

VALID_ESTIMATOR_BASE_TYPE_LOOKUP = BASE_CLASS_LOOKUP
