# -*- coding: utf-8 -*-

__author__ = ["mloning"]
__all__ = ["ESTIMATOR_TEST_PARAMS", "EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sktime.annotation.clasp import ClaSPSegmentation
from sktime.base import BaseEstimator
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.structural import UnobservedComponents
from sktime.registry import (
    BASE_CLASS_LIST,
    BASE_CLASS_LOOKUP,
    ESTIMATOR_TAG_LIST,
    TRANSFORMER_MIXIN_LIST,
)
from sktime.regression.compose import ComposableTimeSeriesForestRegressor
from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel.compose import (
    SeriesToPrimitivesRowTransformer,
    SeriesToSeriesRowTransformer,
)
from sktime.transformations.panel.random_intervals import RandomIntervals
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.transformations.panel.summarize import FittedParamExtractor

# The following estimators currently do not pass all unit tests
# https://github.com/alan-turing-institute/sktime/issues/1627
EXCLUDE_ESTIMATORS = [
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
    # PlateauFinder seems to be broken, see #2259
    "PlateauFinder",
]


EXCLUDED_TESTS = {
    # known issue when X is passed, wrong time indices are returned, #1364
    "StackingForecaster": ["test_predict_time_index_with_X"],
    # known side effects on multivariate arguments, #2072
    "WindowSummarizer": ["test_methods_have_no_side_effects"],
    # test fails in the Panel case for Differencer, see #2522
    "Differencer": ["test_transform_inverse_transform_equivalent"],
    # tagged in issue #2490
    "SignatureClassifier": [
        "test_classifier_on_unit_test_data",
        "test_classifier_on_basic_motions",
    ],
    # test fail with deep problem with pickling inside tensorflow.
    "CNNClassifier": [
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_fit_does_not_overwrite_hyper_params",
    ],
    # pickling problem with local method see #2490
    "ProximityStump": [
        "test_persistence_via_pickle",
        "test_fit_does_not_overwrite_hyper_params",
    ],
    "ProximityTree": [
        "test_persistence_via_pickle",
        "test_fit_does_not_overwrite_hyper_params",
    ],
    "ProximityForest": [
        "test_persistence_via_pickle",
        "test_fit_does_not_overwrite_hyper_params",
    ],
    # sth is not quite right with the RowTransformer-s changing state,
    #   but these are anyway on their path to deprecation, see #2370
    "SeriesToPrimitivesRowTransformer": ["test_methods_do_not_change_state"],
    "SeriesToSeriesRowTransformer": ["test_methods_do_not_change_state"],
    # ColumnTransformer still needs to be refactored, see #2537
    "ColumnTransformer": [
        "test_methods_do_not_change_state",
        "test_fit_transform_output",
    ],
    # Early classifiers intentionally retain information from pervious predict calls
    #   for #1.
    # #2 amd #3 are due to predict/predict_proba returning two items and that breaking
    #   assert_array_equal
    "TEASER": [
        "test_methods_do_not_change_state",
        "test_fit_idempotent",
        "test_persistence_via_pickle",
    ],
    "VARMAX": "test_update_predict_single",  # see 2997, sporadic failure, unknown cause
}

# We here configure estimators for basic unit testing, including setting of
# required hyper-parameters and setting of hyper-parameters for faster training.
SERIES_TO_SERIES_TRANSFORMER = StandardScaler()
SERIES_TO_PRIMITIVES_TRANSFORMER = FunctionTransformer(
    np.mean, kw_args={"axis": 0}, check_inverse=False
)

ESTIMATOR_TEST_PARAMS = {
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
    RandomShapeletTransform: {
        "max_shapelets": 5,
        "n_shapelet_samples": 50,
        "batch_size": 20,
    },
    RandomIntervals: {
        "n_intervals": 3,
    },
    ComposableTimeSeriesForestRegressor: {"n_estimators": 3},
    UnobservedComponents: {"level": "local level"},
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
    # todo: add this back
    # escaping this, since for some estimators
    #   the input format of inverse_transform assumes special col names
    # "inverse_transform",
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
