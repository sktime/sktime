"""Main configuration file for test excludes.

Also contains some other configs, these should be gradually refactored
to registry or to individual tags, where applicable.
"""

__all__ = [
    "EXCLUDE_ESTIMATORS",
    "EXCLUDED_TESTS",
    "MATRIXDESIGN",
    "CYTHON_ESTIMATORS",
    "ONLY_CHANGED_MODULES",
]

from sktime.registry import ESTIMATOR_TAG_LIST

# --------------------
# configs for test run
# --------------------

# whether to subsample estimators per os/version partition matrix design
# default is False, can be set to True by pytest --matrixdesign True flag
MATRIXDESIGN = False

# whether to test only estimators that require cython, C compiler such as gcc
# default is False, can be set to True by pytest --only_cython_estimators True flag
CYTHON_ESTIMATORS = False

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


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
    "MatrixProfileTransformer",
    # tapnet based estimators fail stochastically for unknown reasons, see #3525
    "TapNetRegressor",
    "TapNetClassifier",
    "ResNetClassifier",  # known ResNetClassifier sporafic failures, see #3954
    "LSTMFCNClassifier",  # unknown cause, see bug report #4033
    # DL classifier suspected to cause hangs and memouts, see #4610
    "FCNClassifier",
    "EditDist",
    "CNNClassifier",
    "FCNClassifier",
    "InceptionTimeClassifier",
    "LSTMFCNClassifier",
    "MLPClassifier",
    "MLPRegressor",
    "CNNRegressor",
    "ResNetRegressor",
    "FCNRegressor",
    "LSTMFCNRegressor",
    "CNTCClassifier",
    # splitters excluded with undiagnosed failures, see #6194
    # these are temporarily skipped to allow merging of the base test framework
    "SameLocSplitter",
    "TestPlusTrainSplitter",
    "Repeat",
    "CutoffFhSplitter",
    # sporadic timeouts, see #6344
    "ShapeletLearningClassifierTslearn",
    "DartsXGBModel",
    # models with large weights
    "MomentFMForecaster",
    # Large datasets
    "M5Dataset",
    # Test estimators
    "_TransformChangeNInstances",
    # ptf global models fail the tests, see #7997
    "PytorchForecastingTFT",
    "PytorchForecastingNBeats",
    "PytorchForecastingNHiTS",
    "PytorchForecastingDeepAR",
    # STDBSCAN is not API compliant, see #7994
    "STDBSCAN",
    # Temporarily remove RRF from tests, while #7380 is not merged
    "RecursiveReductionForecaster",
    # DistanceFeatures does ont work for hierarchical data, see #8077
    "DistanceFeatures",
    # TimeSeriesKvisibility is not API compliant, see #8026 and #8072
    "TimeSeriesKvisibility",
    # fails due to #8151 or #8059
    "CNTCRegressor",
    "FreshPRINCE",
    # multiple timeouts and sporadic failures reported related to VARMAX
    # 2997, 3176, 7985
    "VARMAX",
    "SARIMAX",
    "SCINetForecaster",  # known bug #7871
    "MAPAForecaster",  # known bug #8039
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
        "test_multiprocessing_idempotent",  # see 5658
        "test_fit_idempotent",  # see 6637
    ],
    "ProximityForest": [
        "test_persistence_via_pickle",
        "test_fit_does_not_overwrite_hyper_params",
        "test_save_estimators_to_file",
        "test_fit_idempotent",  # see 6201
        "test_multiprocessing_idempotent",  # see 6637
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
    "LSTMFCNRegressor": [
        "test_fit_idempotent",
    ],
    "MLPClassifier": [
        "test_fit_idempotent",
    ],
    "MLPRegressor": [
        "test_fit_idempotent",
    ],
    "CNTCClassifier": [
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "InceptionTimeClassifier": [
        "test_fit_idempotent",
    ],
    "SimpleRNNClassifier": [
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
        "test_multioutput",  # see 6201
        "test_classifier_on_unit_test_data",  # see 6201
    ],
    "SimpleRNNRegressor": [
        "test_fit_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "MCDCNNRegressor": [
        "test_fit_idempotent",
    ],
    "FCNRegressor": [
        "test_fit_idempotent",
    ],
    "InceptionTimeRegressor": [
        "test_fit_idempotent",
    ],
    "CNTCRegressor": [
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
        "test_multiprocessing_idempotent",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "CNNNetwork": "test_inheritance",  # not a registered base class, WiP, see #3028
    "InformationGainSegmentation": [
        "test_inheritance",
        "test_create_test_instance",
    ],
    # SAX returns strange output format
    # this needs to be fixed, was not tested previously due to legacy exception
    "SAXlegacy": ["test_fit_transform_output"],
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
    # networks do not support negative fh
    "HFTransformersForecaster": ["test_predict_time_index_in_sample_full"],
    "PyKANForecaster": ["test_predict_time_index_in_sample_full"],
    "WEASEL": ["test_multiprocessing_idempotent"],  # see 5658
    # StatsForecastMSTL is failing in probabistic forecasts, see #5703, #5920
    "StatsForecastMSTL": ["test_pred_int_tag"],
    # KNeighborsTimeSeriesClassifierTslearn crashes in parallel mode
    "KNeighborsTimeSeriesClassifierTslearn": ["test_multiprocessing_idempotent"],
    # ShapeletTransformPyts creates nested numpy shapelets sporadically, see #6171
    "ShapeletTransformPyts": ["test_non_state_changing_method_contract"],
    "TimeSeriesSVRTslearn": [  # not deterministic, see 6274
        "test_fit_idempotent",
        "test_multiprocessing_idempotent",
    ],
    # ShapeletLearningClassifier is non-pickleable due to DL dependencies
    "ShapeletLearningClassifierTslearn": [
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
        "test_fit_idempotent",
    ],
    "TSRGridSearchCV": ["test_multioutput"],  # see 6708
    # pickling problem
    "ChronosForecaster": [
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "ClusterSegmenter": [
        "test_doctest_examples",
        "test_predict_points",
        "test_predict_segments",
        "test_transform_output_type",
        "test_output_type",
    ],
    "GreedyGaussianSegmentation": [
        "test_predict_points",
        "test_predict_segments",
        "test_output_type",
        "test_transform_output_type",
        "test_inheritance",
        "test_create_test_instance",
    ],
    # see PR 7921
    "RocketClassifier": ["test_classifier_on_basic_motions"],
    # see bug report #6465 and #7958
    "MACNNClassifier": [
        "test_multioutput",
        "test_classifier_on_unit_test_data",
    ],
    "MCDCNNClassifier": [
        "test_persistence_via_pickle",
        "test_multioutput",
        "test_classifier_on_unit_test_data",
        "test_fit_idempotent",  # not part of bug reports but due to randomness
    ],
    "ARLagOrderSelector": [
        "test_doctest_examples",  # doctest fails, see #8129
    ],
    "ESRNNForecaster": [  # pickling problem, see #8135
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "TSFreshClassifier": ["test_multiprocessing_idempotent"],  # see 8150
    "MomentFMClassifier": [
        "test_fit_idempotent",
        "test_multiprocessing_idempotent",
    ],  # see 8253
}

# exclude tests but keyed by test name
EXCLUDED_TESTS_BY_TEST = {
    "test_get_test_params_coverage": [
        "BOSSEnsemble",
        "CAPA",
        "CNTCClassifier",
        "CNTCNetwork",
        "CanonicalIntervalForest",
        "CircularBinarySegmentation",
        "ClaSPTransformer",
        "ClearSky",
        "ColumnEnsembleClassifier",
        "ColumnwiseTransformer",
        "ContractableBOSS",
        "DOBIN",
        "DWTTransformer",
        "DilationMappingTransformer",
        "DirRecTabularRegressionForecaster",
        "DirRecTimeSeriesRegressionForecaster",
        "DirectTimeSeriesRegressionForecaster",
        "DistFromAligner",
        "DistanceFeatures",
        "DontUpdate",
        "DummyRegressor",
        "ElasticEnsemble",
        "FeatureSelection",
        "FreshPRINCE",
        "GreedyGaussianSegmentation",
        "HCrystalBallAdapter",
        "HIVECOTEV1",
        "HIVECOTEV2",
        "Hidalgo",
        "HierarchicalProphet",
        "InceptionTimeNetwork",
        "IndividualBOSS",
        "IndividualTDE",
        "InformationGainSegmentation",
        "LTSFDLinearForecaster",
        "LTSFLinearForecaster",
        "LTSFNLinearForecaster",
        "LogTransformer",
        "M5Dataset",
        "MCDCNNClassifier",
        "MCDCNNNetwork",
        "MCDCNNRegressor",
        "MLPNetwork",
        "MUSE",
        "MVCAPA",
        "MatrixProfile",
        "MatrixProfileTransformer",
        "MiniRocketMultivariate",
        "MiniRocketMultivariateVariable",
        "MovingWindow",
        "MultiRocket",
        "MultioutputTabularRegressionForecaster",
        "MultioutputTimeSeriesRegressionForecaster",
        "OnlineEnsembleForecaster",
        "OptionalPassthrough",
        "PAA",
        "PAAlegacy",
        "PCATransformer",
        "PELT",
        "PaddingTransformer",
        "PlateauFinder",
        "Prophet",
        "ProphetPiecewiseLinearTrendForecaster",
        "Prophetverse",
        "RandomIntervalClassifier",
        "RandomIntervalFeatureExtractor",
        "RandomIntervalSegmenter",
        "RandomIntervalSpectralEnsemble",
        "RandomIntervals",
        "RandomSamplesAugmenter",
        "RandomShapeletTransform",
        "RecursiveTabularRegressionForecaster",
        "RecursiveTimeSeriesRegressionForecaster",
        "ReducerTransform",
        "SAX",
        "SAXlegacy",
        "SFA",
        "SFAFast",
        "SeededBinarySegmentation",
        "ShapeletTransform",
        "ShapeletTransformClassifier",
        "SignatureClassifier",
        "SignatureTransformer",
        "SlidingWindowSegmenter",
        "SlopeTransformer",
        "StackingForecaster",
        "StatThresholdAnomaliser",
        "SummaryClassifier",
        "SupervisedTimeSeriesForest",
        "TEASER",
        "TSFreshClassifier",
        "TapNetNetwork",
        "TemporalDictionaryEnsemble",
        "TimeBinner",
        "TimeSeriesKMedoids",
        "TimeSeriesKernelKMeans",
        "TruncationTransformer",
        "UnobservedComponents",
        "WEASEL",
        "WhiteNoiseAugmenter",
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
        "GeometricMeanAbsoluteError",
        "MedianRelativeAbsoluteError",
        "MeanSquaredScaledError",
        "GeometricMeanRelativeAbsoluteError",
        "GeometricMeanRelativeSquaredError",
        "MedianSquaredPercentageError",
        "MedianAbsoluteScaledError",
        "MedianSquaredError",
        "MeanAbsolutePercentageError",
        "MeanAbsoluteScaledError",
        "MeanAbsoluteError",
        "MedianAbsoluteError",
        "MeanSquaredPercentageError",
        "MedianAbsolutePercentageError",
        "MeanSquaredError",
        "PinballLoss",
        "RelativeLoss",
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
