"""Main configuration file for test excludes.

Also contains some other configs, these should be gradually refactored
to registry or to individual tags, where applicable.
"""

__all__ = [
    "EXCLUDE_ESTIMATORS",
    "EXCLUDED_TESTS",
    "MATRIXDESIGN",
    "ONLY_CHANGED_MODULES",
    "ONLY_VM_ESTIMATORS",
]

from sktime.registry import ESTIMATOR_TAG_LIST

# --------------------
# configs for test run
# --------------------

# whether to subsample estimators per os/version partition matrix design
# default is False, can be set to True by pytest --matrixdesign True flag
MATRIXDESIGN = False

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False

# whether to test only estimators from modules require a VM to test
# default is False, can be set to True by pytest --only_vm_estimators True flag
ONLY_VM_ESTIMATORS = False


# DO NOT ADD ESTIMATORS HERE ANYMORE
# ADD TEST SKIPS TO TAG tag tests:skip_all INSTEAD
EXCLUDE_ESTIMATORS = []

# DO NOT ADD ESTIMATORS HERE ANYMORE
# ADD TEST SKIPS TO TAG tag tests:skip_by_name INSTEAD
EXCLUDED_TESTS = {}

# DO NOT ADD ESTIMATORS HERE ANYMORE
# ADD TEST SKIPS TO TAG tag tests:skip_by_name INSTEAD
# exclude tests but keyed by test name
EXCLUDED_TESTS_BY_TEST = {
    "test_get_test_params_coverage": [
        "CNTCNetwork",
        "ClaSPTransformer",
        "ClearSky",
        "ContractableBOSS",
        "DOBIN",
        "DilationMappingTransformer",
        "DirRecTabularRegressionForecaster",
        "DirRecTimeSeriesRegressionForecaster",
        "DirectTimeSeriesRegressionForecaster",
        "DistFromAligner",
        "DistanceFeatures",
        "DummyRegressor",
        "ElasticEnsemble",
        "FeatureSelection",
        "FreshPRINCE",
        "HCrystalBallAdapter",
        "HIVECOTEV1",
        "HIVECOTEV2",
        "Hidalgo",
        "HierarchicalProphet",
        "InceptionTimeNetwork",
        "IndividualBOSS",
        "IndividualTDE",
        "M5Dataset",
        "MCDCNNClassifier",
        "MCDCNNNetwork",
        "MCDCNNRegressor",
        "MLPNetwork",
        "MUSE",
        "MultioutputTabularRegressionForecaster",
        "MultioutputTimeSeriesRegressionForecaster",
        "OnlineEnsembleForecaster",
        "PAAlegacy",
        "Prophetverse",
        "RandomIntervalClassifier",
        "RandomIntervalFeatureExtractor",
        "RandomIntervalSegmenter",
        "RandomIntervalSpectralEnsemble",
        "RandomSamplesAugmenter",
        "RecursiveTabularRegressionForecaster",
        "RecursiveTimeSeriesRegressionForecaster",
        "SAXlegacy",
        "SFA",
        "SFAFast",
        "ShapeletTransform",
        "ShapeletTransformClassifier",
        "SlidingWindowSegmenter",
        "StackingForecaster",
        "SummaryClassifier",
        "SupervisedTimeSeriesForest",
        "TEASER",
        "TSFreshClassifier",
        "TapNetNetwork",
        "TemporalDictionaryEnsemble",
        "TimeSeriesKMedoids",
        "WEASEL",
        # The below estimators need to have their name removed from EXCLUDE_SOFT_DEPS
        # too after adding test parameters to them
        "BaggingForecaster",
        "ClustererPipeline",
        "DirectTabularRegressionForecaster",
        "EnbPIForecaster",
        "FittedParamExtractor",
        "ForecastingOptunaSearchCV",
        "HFTransformersForecaster",
        "HolidayFeatures",
        "ParamFitterPipeline",
        "PluginParamsForecaster",
        "PluginParamsTransformer",
        "RegressorPipeline",
        "SupervisedIntervals",
        "TSBootstrapAdapter",
        "ThetaModularForecaster",
        "WeightedEnsembleClassifier",
    ],
    "test_doctest_examples": [
        # between-versions inconsistency how doctest handles np.float64.
        # on lower version, prints 0.123456
        # on higher version, prints np.float64(0.123456)
        # therefore these doctests will fail either on lower or higher versions
        "MedianSquaredScaledError",
        "RMSEnormalizedByIQR",
        "KLDivergenceDoubleExponential",
        "KLDivergenceNormal",
        "KLDivergenceSingleExponential",
        "MSEnormalizedBySD",
        "GeometricMeanAbsoluteError",
        "MedianRelativeAbsoluteError",
        "MeanSquaredScaledError",
        "GeometricMeanRelativeAbsoluteError",
        "GeometricMeanRelativeSquaredError",
        "MedianSquaredPercentageError",
        "MedianAbsoluteScaledError",
        "MedianSquaredError",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanAbsolutePercentageErrorStabilized",
        "MeanAbsoluteScaledError",
        "MedianAbsoluteError",
        "MeanSquaredPercentageError",
        "MedianAbsolutePercentageError",
        "MeanSquaredError",
        "PinballLoss",
        "RelativeLoss",
        "TheilU2",
        "MeanRelativeAbsoluteError",
    ],
}

# estimators that have 2 test params only when their soft dependency is installed
EXCLUDE_SOFT_DEPS = [
    "BaggingForecaster",
    "ClustererPipeline",
    "DirectTabularRegressionForecaster",
    "EnbPIForecaster",
    "FittedParamExtractor",
    "ForecastingOptunaSearchCV",
    "HFTransformersForecaster",
    "HolidayFeatures",
    "ParamFitterPipeline",
    "PluginParamsForecaster",
    "PluginParamsTransformer",
    "RegressorPipeline",
    "SupervisedIntervals",
    "TSBootstrapAdapter",
    "ThetaModularForecaster",
    "WeightedEnsembleClassifier",
]

# add EXCLUDED_TESTS_BY_TEST to EXCLUDED_TESTS
# the latter is the single source of truth
for k, v in EXCLUDED_TESTS_BY_TEST.items():
    for est in v:
        EXCLUDED_TESTS.setdefault(est, []).extend([k])

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
