__author__ = ["mloning"]
__all__ = ["EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

from sktime.base import BaseEstimator, BaseObject
from sktime.registry import (
    BASE_CLASS_LIST,
    BASE_CLASS_LOOKUP,
    ESTIMATOR_TAG_LIST,
    TRANSFORMER_MIXIN_LIST,
)
from sktime.transformations.base import BaseTransformer

EXCLUDE_ESTIMATORS = [
    # SFA is non-compliant with any transformer interfaces, #2064
    "SFA",
    # PlateauFinder seems to be broken, see #2259
    "PlateauFinder",
    # below are removed due to mac failures we don't fully understand, see #3103
    "HIVECOTEV1",
    "HIVECOTEV2",
    "RandomIntervalSpectralEnsemble",
    "RandomInvervals",
    "RandomIntervalSegmenter",
    "RandomIntervalFeatureExtractor",
    "RandomIntervalClassifier",
    "MiniRocket",
    "MatrixProfileTransformer",
    # tapnet based estimators fail stochastically for unknown reasons, see #3525
    "TapNetRegressor",
    "TapNetClassifier",
    "ResNetClassifier",  # known ResNetClassifier sporafic failures, see #3954
    "LSTMFCNClassifier",  # unknown cause, see bug report #4033
    "TimeSeriesLloyds",  # an abstract class, but does not follow naming convention
    # DL classifier suspected to cause hangs and memouts, see #4610
    "FCNClassifier",
    "MACNNClassifier",
    "SimpleRNNClassifier",
    "SimpleRNNRegressor",
    "EditDist",
    "CNNClassifier",
    "FCNClassifier",
    "InceptionTimeClassifer",
    "LSTMFCNClassifier",
    "MLPClassifier",
    "CNNRegressor",
    "ResNetRegressor",
]


EXCLUDED_TESTS = {
    # issue when prediction intervals, see #3479 and #4504
    # known issue with prediction intervals that needs fixing, tracked in #4181
    "SquaringResiduals": [
        "test_predict_time_index",
        "test_predict_residuals",
        "test_predict_interval",
        "test_predict_time_index_with_X",  # separate - refer to #4765
    ],
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
    # pickling problem with local method see #2490
    "ProximityStump": [
        "test_persistence_via_pickle",
        "test_fit_does_not_overwrite_hyper_params",
        "test_save_estimators_to_file",
    ],
    "ProximityTree": [
        "test_persistence_via_pickle",
        "test_fit_does_not_overwrite_hyper_params",
        "test_save_estimators_to_file",
    ],
    "ProximityForest": [
        "test_persistence_via_pickle",
        "test_fit_does_not_overwrite_hyper_params",
        "test_save_estimators_to_file",
    ],
    # TapNet fails due to Lambda layer, see #3539 and #3616
    "TapNetClassifier": [
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "TapNetRegressor": [
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    # `test_fit_idempotent` fails with `AssertionError`, see #3616
    "ResNetClassifier": [
        "test_fit_idempotent",
    ],
    "ResNetRegressor": [
        "test_fit_idempotent",
    ],
    "CNNClassifier": [
        "test_fit_idempotent",
    ],
    "CNNRegressor": [
        "test_fit_idempotent",
    ],
    "FCNClassifier": [
        "test_fit_idempotent",
    ],
    "LSTMFCNClassifier": [
        "test_fit_idempotent",
    ],
    "MLPClassifier": [
        "test_fit_idempotent",
    ],
    "InceptionTimeClassifier": [
        "test_fit_idempotent",
    ],
    "SimpleRNNClassifier": [
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "SimpleRNNRegressor": [
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "MCDCNNClassifier": [
        "test_fit_idempotent",
    ],
    "MCDCNNRegressor": [
        "test_fit_idempotent",
    ],
    "MACNNClassifier": [
        "test_fit_idempotent",
    ],
    # sth is not quite right with the RowTransformer-s changing state,
    #   but these are anyway on their path to deprecation, see #2370
    "SeriesToPrimitivesRowTransformer": ["test_methods_do_not_change_state"],
    "SeriesToSeriesRowTransformer": ["test_methods_do_not_change_state"],
    # ColumnTransformer still needs to be refactored, see #2537
    "ColumnTransformer": ["test_methods_do_not_change_state"],
    # Early classifiers intentionally retain information from previous predict calls
    #   for #1.
    # #2 amd #3 are due to predict/predict_proba returning two items and that breaking
    #   assert_array_equal
    "TEASER": [
        "test_non_state_changing_method_contract",
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "CNNNetwork": "test_inheritance",  # not a registered base class, WiP, see #3028
    "VARMAX": [
        "test_update_predict_single",  # see 2997, sporadic failure, unknown cause
        "test__y_when_refitting",  # see 3176
    ],
    # GGS inherits from BaseEstimator which breaks this test
    "GreedyGaussianSegmentation": ["test_inheritance", "test_create_test_instance"],
    "InformationGainSegmentation": [
        "test_inheritance",
        "test_create_test_instance",
    ],
    # SAX returns strange output format
    # this needs to be fixed, was not tested previously due to legacy exception
    "SAX": "test_fit_transform_output",
    "DynamicFactor": [
        "test_predict_time_index_in_sample_full",  # refer to #4765
    ],
    "ARIMA": [
        "test_predict_time_index_in_sample_full",  # refer to #4765
    ],
    "VECM": [
        "test_hierarchical_with_exogeneous",  # refer to #4743
    ],
    "Pipeline": ["test_inheritance"],  # does not inherit from intermediate base classes
}

# We use estimator tags in addition to class hierarchies to further distinguish
# estimators into different categories. This is useful for defining and running
# common tests for estimators with the same tags.
VALID_ESTIMATOR_TAGS = tuple(ESTIMATOR_TAG_LIST)

# NON_STATE_CHANGING_METHODS =
# methods that should not change the state of the estimator, that is, they should
# not change fitted parameters or hyper-parameters. They are also the methods that
# "apply" the fitted estimator to data and useful for checking results.
# NON_STATE_CHANGING_METHODS_ARRAYLIK =
# non-state-changing methods that return an array-like output

NON_STATE_CHANGING_METHODS_ARRAYLIKE = (
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

NON_STATE_CHANGING_METHODS = NON_STATE_CHANGING_METHODS_ARRAYLIKE + (
    "get_fitted_params",
)

# The following gives a list of valid estimator base classes.
VALID_TRANSFORMER_TYPES = tuple(TRANSFORMER_MIXIN_LIST) + (BaseTransformer,)

BASE_BASE_TYPES = (BaseEstimator, BaseObject)
VALID_ESTIMATOR_BASE_TYPES = tuple(set(BASE_CLASS_LIST).difference(BASE_BASE_TYPES))

VALID_ESTIMATOR_TYPES = (
    BaseEstimator,
    *VALID_ESTIMATOR_BASE_TYPES,
    *VALID_TRANSFORMER_TYPES,
)

VALID_ESTIMATOR_BASE_TYPE_LOOKUP = BASE_CLASS_LOOKUP
