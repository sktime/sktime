#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["check_estimator"]

import numbers
import types
from inspect import signature

import joblib
import numpy as np
from sklearn import clone
from sklearn.utils.estimator_checks import \
    check_get_params_invariance as _check_get_params_invariance
from sklearn.utils.estimator_checks import \
    check_set_params as _check_set_params
from sktime.base import BaseEstimator
from sktime.classification.base import BaseClassifier
from sktime.forecasting.base import BaseForecaster
from sktime.regression.base import BaseRegressor
from sktime.transformers.base import BaseTransformer
from sktime.utils.testing.construct import _construct_instance
from sktime.utils.testing.inspect import _get_args

IMPLEMENTED_BASE_ESTIMATORS = [
    BaseClassifier,
    BaseRegressor,
    BaseForecaster,
    BaseTransformer
]


def yield_estimator_checks():
    checks = [
        check_has_common_interface,
        check_get_params,
        check_clone,
    ]
    for check in checks:
        yield check


def check_meta_estimators(Estimator):
    if hasattr(Estimator, "_required_parameters"):
        params = Estimator._required_parameters
        assert isinstance(params, list)
        assert all([isinstance(param, str) for param in params])


def check_inheritance(Estimator):
    assert issubclass(Estimator, BaseEstimator)

    # check that inherits from one and only one task-specific estimator
    assert sum(issubclass(Estimator, base_estimator)
               for base_estimator in IMPLEMENTED_BASE_ESTIMATORS) == 1


def check_has_common_interface(Estimator):
    assert hasattr(Estimator, "is_fitted")
    assert isinstance(Estimator.is_fitted, property)

    # check instance
    estimator = _construct_instance(Estimator)
    methods = [
        "fit",
        "check_is_fitted",
        "is_fitted",
        "_is_fitted",
        "set_params",
        "get_params"
    ]
    for method in methods:
        assert hasattr(estimator, method)
    assert (hasattr(estimator, "predict") or hasattr(estimator, "transform"))


def check_get_params(Estimator):
    estimator = _construct_instance(Estimator)
    params = estimator.get_params()
    assert isinstance(params, dict)
    _check_get_params_invariance(estimator.__class__.__name__, estimator)


def check_set_params(Estimator):
    # check set_params returns self
    estimator = _construct_instance(Estimator)
    params = estimator.get_params()
    assert estimator.set_params(**params) is estimator
    _check_set_params(estimator.__class__.__name__, estimator)


def check_clone(Estimator):
    estimator = _construct_instance(Estimator)
    clone(estimator)


def check_repr(Estimator):
    estimator = _construct_instance(Estimator)
    repr(estimator)


def check_constructor(Estimator):
    estimator = _construct_instance(Estimator)

    # Check that init does not construct object of other class than itself
    assert isinstance(estimator, Estimator)

    # Ensure that each parameter is set in init
    init_params = _get_args(type(estimator).__init__)
    invalid_attr = set(init_params) - set(vars(estimator)) - {"self"}
    assert not invalid_attr, (
            "Estimator %s should store all parameters"
            " as an attribute during init. Did not find "
            "attributes %s."
            % (estimator.__class__.__name__, sorted(invalid_attr)))

    # Ensure that init does nothing but set parameters
    # No logic/interaction with other parameters
    def param_filter(p):
        """Identify hyper parameters of an estimator"""
        return (p.name != 'self' and
                p.kind != p.VAR_KEYWORD and
                p.kind != p.VAR_POSITIONAL)

    init_params = [p for p in signature(estimator.__init__).parameters.values()
                   if param_filter(p)]

    params = estimator.get_params()

    required_params = getattr(estimator, '_required_parameters', [])
    init_params = [param for param in init_params if
                   param not in required_params]

    for param in init_params:
        assert param.default != param.empty, (
                "parameter %s for %s has no default value"
                % (param.name, estimator.__class__.__name__))
        if type(param.default) is type:
            assert param.default in [np.float64, np.int64]
        else:
            assert (type(param.default) in
                    [str, int, float, bool, tuple, type(None),
                     np.float64, types.FunctionType, joblib.Memory])

        param_value = params[param.name]
        if isinstance(param_value, np.ndarray):
            np.testing.assert_array_equal(param_value, param.default)
        else:
            if bool(isinstance(param_value, numbers.Real) and np.isnan(
                    param_value)):
                # Allows to set default parameters to np.nan
                assert param_value is param.default, param.name
            else:
                assert param_value == param.default, param.name


# def _fit(estimator):
#     if isinstance(estimator, BaseForecaster):
#         y_train, _, fh = make_forecasting_problem()
#         return estimator.fit(y, fh)
#
#     elif isinstance(estimator, BaseClassifier):
#         X, y = make_classification_problem()
#         return estimator.fit(X, y)
#
#     elif isinstance(estimator, BaseRegressor):
#         X, y = make_regression_problem()
#         return estimator.fit(X, y)
#
#     else:
#         raise ValueError("estimator type not supported")


def _predict(estimator):
    pass


def check_estimator(Estimator):
    for check in yield_estimator_checks():
        check(Estimator)