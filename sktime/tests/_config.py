# -*- coding: utf-8 -*-

__author__ = ["mloning"]
__all__ = ["ESTIMATOR_TEST_PARAMS", "EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

import numpy as np
from pyod.models.knn import KNN
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sktime.annotation.adapters import PyODAnnotator
from sktime.annotation.clasp import ClaSPSegmentation
from sktime.base import BaseEstimator
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
from sktime.transformations.panel.interpolate import TSInterpolator
from sktime.transformations.panel.random_intervals import RandomIntervals
from sktime.transformations.panel.reduce import Tabularizer
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.transformations.panel.summarize import FittedParamExtractor
from sktime.transformations.series.adapt import TabularToSeriesAdaptor

# The following estimators currently do not pass all unit tests
# https://github.com/alan-turing-institute/sktime/issues/1627
EXCLUDE_ESTIMATORS = [
    # known issues, see PR 1989 for fix
    "ProximityForest",
    "ProximityStump",
    "ProximityTree",
    # ConditionalDeseasonalizer and STLtransformer still need refactoring
    #  (see PR 1773, blocked through open discussion) escaping until then
    "ConditionalDeseasonalizer",
    "STLTransformer",
    # SFA is non-compliant with any transformer interfaces, #2064
    "SFA",
    # requires y in fit, this is incompatible with the old testing framework
    #    unless it inherits from the old mixins, which hard coded the y
    #    should be removed once test_all_transformers has been refactored to scenarios
    "TSFreshRelevantFeatureExtractor",
]


EXCLUDED_TESTS = {
    # known issue when X is passed, wrong time indices are returned, #1364
    "StackingForecaster": ["test_predict_time_index_with_X"],
    # known side effects on multivariate arguments, #2072
    "WindowSummarizer": ["test_methods_have_no_side_effects"],
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
    RandomShapeletTransform: {
        "max_shapelets": 5,
        "n_shapelet_samples": 50,
        "batch_size": 20,
    },
    RandomIntervals: {
        "n_intervals": 3,
    },
    TSInterpolator: {"length": 10},
    ComposableTimeSeriesForestRegressor: {"n_estimators": 3},
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
    "predict_var",
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
