# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Registry and dispatcher for test classes.

Module does not contain tests, only test utilities.
"""

__author__ = ["fkiraly"]

from inspect import isclass


def get_test_class_registry():
    """Return test class registry.

    Wrapped in a function to avoid circular imports.

    Returns
    -------
    testclass_dict : dict
        test class registry
        keys are scitypes, values are test classes TestAll[Scitype]
    """
    from sktime.alignment.tests.test_all_aligners import TestAllAligners
    from sktime.annotation.tests.test_all_annotators import TestAllAnnotators
    from sktime.classification.early_classification.tests.test_all_early_classifiers import (  # noqa E501
        TestAllEarlyClassifiers,
    )
    from sktime.classification.tests.test_all_classifiers import TestAllClassifiers
    from sktime.dists_kernels.tests.test_all_dist_kernels import (
        TestAllPairwiseTransformers,
        TestAllPanelTransformers,
    )
    from sktime.forecasting.tests.test_all_forecasters import (
        TestAllForecasters,
        TestAllGlobalForecasters,
    )
    from sktime.param_est.tests.test_all_param_est import TestAllParamFitters
    from sktime.proba.tests.test_all_distrs import TestAllDistributions
    from sktime.regression.tests.test_all_regressors import TestAllRegressors
    from sktime.split.tests.test_all_splitters import TestAllSplitters
    from sktime.tests.test_all_estimators import TestAllEstimators, TestAllObjects
    from sktime.transformations.tests.test_all_transformers import TestAllTransformers

    testclass_dict = dict()
    # every object in sktime inherits from BaseObject
    # "object" tests are run for all objects
    testclass_dict["object"] = TestAllObjects
    # fittable objects inherit from BaseEstimator
    # "estimator" tests are run for all estimators
    # estimators are also objects
    testclass_dict["estimator"] = TestAllEstimators
    # more specific base classes
    # these inherit either from BaseEstimator or BaseObject,
    # so also imply estimator and object tests, or only object tests
    testclass_dict["aligner"] = TestAllAligners
    testclass_dict["classifier"] = TestAllClassifiers
    testclass_dict["distribution"] = TestAllDistributions
    testclass_dict["early_classifier"] = TestAllEarlyClassifiers
    testclass_dict["forecaster"] = TestAllForecasters
    testclass_dict["global_forecaster"] = TestAllGlobalForecasters
    testclass_dict["param_est"] = TestAllParamFitters
    testclass_dict["regressor"] = TestAllRegressors
    testclass_dict["series-annotator"] = TestAllAnnotators
    testclass_dict["splitter"] = TestAllSplitters
    testclass_dict["transformer"] = TestAllTransformers
    testclass_dict["transformer-pairwise"] = TestAllPairwiseTransformers
    testclass_dict["transformer-pairwise-panel"] = TestAllPanelTransformers

    return testclass_dict


def get_test_classes_for_obj(obj):
    """Get all test classes relevant for an object or estimator.

    Parameters
    ----------
    obj : object or estimator, descendant of sktime BaseObject or BaseEstimator
        object or estimator for which to get test classes

    Returns
    -------
    test_classes : list of test classes
        list of test classes relevant for obj
        these are references to the actual classes, not strings
        if obj was not a descendant of BaseObject or BaseEstimator, returns empty list
    """
    from sktime.base import BaseEstimator, BaseObject
    from sktime.registry import scitype

    def is_object(obj):
        """Return whether obj is an estimator class or estimator object."""
        if isclass(obj):
            return issubclass(obj, BaseObject)
        else:
            return isinstance(obj, BaseObject)

    def is_estimator(obj):
        """Return whether obj is an estimator class or estimator object."""
        if isclass(obj):
            return issubclass(obj, BaseEstimator)
        else:
            return isinstance(obj, BaseEstimator)

    if not is_object(obj):
        return []

    testclass_dict = get_test_class_registry()

    # we always need to run "object" tests
    test_clss = [testclass_dict["object"]]

    if is_estimator(obj):
        test_clss += [testclass_dict["estimator"]]

    try:
        obj_scitypes = scitype(obj, force_single_scitype=False, coerce_to_list=True)
    except Exception:
        obj_scitypes = []

    for obj_scitype in obj_scitypes:
        if obj_scitype in testclass_dict:
            test_clss += [testclass_dict[obj_scitype]]

    return test_clss
