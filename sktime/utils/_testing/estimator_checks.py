#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# adapted from scikit-learn's estimator_checks

__author__ = ["Markus Löning"]
__all__ = ["check_estimator"]

import numbers
import pickle
import types
from copy import deepcopy
from inspect import signature

import joblib
import numpy as np
import pytest
from sklearn import clone
from sklearn.utils.estimator_checks import (
    check_get_params_invariance as _check_get_params_invariance,
)
from sklearn.utils.estimator_checks import check_set_params as _check_set_params
from sklearn.utils.testing import set_random_state

from sktime.base import BaseEstimator
from sktime.classification.base import BaseClassifier
from sktime.classification.base import is_classifier
from sktime.exceptions import NotFittedError
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._base import is_forecaster
from sktime.regression.base import BaseRegressor
from sktime.regression.base import is_regressor
from sktime.tests._config import NON_STATE_CHANGING_METHODS
from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.base import (
    is_non_fittable_series_as_features_transformer,
)
from sktime.transformers.series_as_features.base import (
    is_series_as_features_transformer,
)
from sktime.transformers.single_series.base import BaseSingleSeriesTransformer
from sktime.transformers.single_series.base import is_single_series_transformer
from sktime.utils._testing import ESTIMATOR_TEST_PARAMS
from sktime.utils._testing import _assert_array_almost_equal
from sktime.utils._testing import _assert_array_equal
from sktime.utils._testing import _construct_instance
from sktime.utils._testing import _get_args
from sktime.utils._testing import _make_args


def check_estimator(Estimator, exclude=None):
    """Check whether estimator complies with common interface.

    Parameters
    ----------
    Estimator : Estimator class

    Raises
    ------
    AssertionError
        If Estimator does not comply
    """
    for check in yield_estimator_checks(exclude=exclude):
        check(Estimator)


def yield_estimator_checks(exclude=None):
    """Iterator to yield estimator checks"""
    checks = [
        check_inheritance,
        check_required_params,
        check_has_common_interface,
        check_constructor,
        check_get_params,
        check_set_params,
        check_clone,
        check_repr,
        check_fit_updates_state,
        check_fit_returns_self,
        check_raises_not_fitted_error,
        check_fit_idempotent,
        check_fit_does_not_overwrite_hyper_params,
        check_methods_do_not_change_state,
        check_persistence_via_pickle,
        check_multiprocessing_idempotent,
    ]
    for check in checks:
        # check if associated test is not included in the exclusion list
        if check.__name__ in exclude:
            continue
        yield check


def check_required_params(Estimator):
    # Check common meta-estimator interface
    if hasattr(Estimator, "_required_parameters"):
        required_params = Estimator._required_parameters

        assert isinstance(required_params, list), (
            f"For estimator: {Estimator}, `_required_parameters` must be a "
            f"list, but found type: {type(required_params)}"
        )

        assert all([isinstance(param, str) for param in required_params]), (
            f"For estimator: {Estimator}, elements of `_required_parameters` "
            f"list must be strings"
        )

        # check if needless parameters are in _required_parameters
        init_params = [
            param.name for param in signature(Estimator.__init__).parameters.values()
        ]
        in_required_but_not_init = [
            param for param in required_params if param not in init_params
        ]
        if len(in_required_but_not_init) > 0:
            raise ValueError(
                f"Found parameters in `_required_parameters` which "
                f"are not in `__init__`: "
                f"{in_required_but_not_init}"
            )


def check_inheritance(Estimator):
    # Check that estimator inherits from one and only one task-specific
    # estimator

    # Class checks
    base_classes = [
        BaseClassifier,
        BaseRegressor,
        BaseForecaster,
        BaseSeriesAsFeaturesTransformer,
        BaseSingleSeriesTransformer,
    ]
    assert issubclass(Estimator, BaseEstimator), (
        f"Estimator: {Estimator} " f"is not a sub-class of " f"BaseEstimator."
    )
    assert (
        sum([issubclass(Estimator, base_class) for base_class in base_classes]) == 1
    ), (
        f"Estimator: {Estimator} is a "
        f"sub-class of more than one "
        f"task-specific base estimators."
    )

    # Instance type checks
    is_type_checks = [
        is_classifier,
        is_regressor,
        is_forecaster,
        is_series_as_features_transformer,
        is_single_series_transformer,
    ]
    estimator = _construct_instance(Estimator)
    assert isinstance(estimator, BaseEstimator), (
        f"Estimator: {estimator.__class__.__name__} "
        f"does not an instance of BaseEstimator."
    )
    assert sum([is_type_check(estimator) for is_type_check in is_type_checks]) == 1, (
        f"Estimator: {estimator.__class__.__name__} is an instance of more "
        f"than one task-specific base estimators."
    )


def check_has_common_interface(Estimator):
    # Check estimator implements the common interface

    # Check class for type of attribute
    assert isinstance(Estimator.is_fitted, property)

    # Check instance
    estimator = _construct_instance(Estimator)
    common_attrs = [
        "fit",
        "check_is_fitted",
        "is_fitted",  # read-only property
        "_is_fitted",  # underlying estimator state
        "set_params",
        "get_params",
    ]
    for attr in common_attrs:
        assert hasattr(estimator, attr), (
            f"Estimator: {estimator.__class__.__name__} does not implement "
            f"attribute: {attr}"
        )
    assert hasattr(estimator, "predict") or hasattr(estimator, "transform")


def check_get_params(Estimator):
    # Check get params works correctly
    estimator = _construct_instance(Estimator)
    params = estimator.get_params()
    assert isinstance(params, dict)
    _check_get_params_invariance(estimator.__class__.__name__, estimator)


def check_set_params(Estimator):
    # Check set_params works correctly
    estimator = _construct_instance(Estimator)
    params = estimator.get_params()
    assert estimator.set_params(**params) is estimator
    _check_set_params(estimator.__class__.__name__, estimator)


def check_clone(Estimator):
    # Check we can call clone from scikit-learn
    estimator = _construct_instance(Estimator)
    clone(estimator)


def check_repr(Estimator):
    # Check we can call repr
    estimator = _construct_instance(Estimator)
    repr(estimator)


def check_constructor(Estimator):
    # Check that the constructor behaves correctly
    estimator = _construct_instance(Estimator)

    # Check that init does not construct object of other class than itself
    assert isinstance(estimator, Estimator)

    # Ensure that each parameter is set in init
    init_params = _get_args(type(estimator).__init__)
    invalid_attr = set(init_params) - set(vars(estimator)) - {"self"}
    assert not invalid_attr, (
        "Estimator %s should store all parameters"
        " as an attribute during init. Did not find "
        "attributes `%s`." % (estimator.__class__.__name__, sorted(invalid_attr))
    )

    # Ensure that init does nothing but set parameters
    # No logic/interaction with other parameters
    def param_filter(p):
        """Identify hyper parameters of an estimator"""
        return (
            p.name != "self" and p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL
        )

    init_params = [
        p for p in signature(estimator.__init__).parameters.values() if param_filter(p)
    ]

    params = estimator.get_params()

    # Filter out required parameters with no default value and parameters
    # set for running tests
    required_params = getattr(estimator, "_required_parameters", [])
    test_config_params = ESTIMATOR_TEST_PARAMS.get(Estimator, {}).keys()

    init_params = [
        param
        for param in init_params
        if param.name not in required_params and param.name not in test_config_params
    ]

    for param in init_params:
        assert param.default != param.empty, (
            "parameter `%s` for %s has no default value and is not "
            "included in `_required_parameters`"
            % (param.name, estimator.__class__.__name__)
        )
        if type(param.default) is type:
            assert param.default in [np.float64, np.int64]
        else:
            assert type(param.default) in [
                str,
                int,
                float,
                bool,
                tuple,
                type(None),
                np.float64,
                types.FunctionType,
                joblib.Memory,
            ]

        param_value = params[param.name]
        if isinstance(param_value, np.ndarray):
            np.testing.assert_array_equal(param_value, param.default)
        else:
            if bool(isinstance(param_value, numbers.Real) and np.isnan(param_value)):
                # Allows to set default parameters to np.nan
                assert param_value is param.default, param.name
            else:
                assert param_value == param.default, param.name


def check_fit_updates_state(Estimator):
    # Check that fit updates the is-fitted states
    attrs = ["_is_fitted", "is_fitted"]

    estimator = _construct_instance(Estimator)
    # Check it's not fitted before calling fit
    for attr in attrs:
        assert not getattr(
            estimator, attr
        ), f"Estimator: {estimator} does not initiate attribute: {attr} to False"

    fit_args = _make_args(estimator, "fit")
    estimator.fit(*fit_args)

    # Check states are updated after calling fit
    for attr in attrs:
        assert getattr(
            estimator, attr
        ), f"Estimator: {estimator} does not update attribute: {attr} during fit"


def check_fit_returns_self(Estimator):
    # Check that fit returns self
    estimator = _construct_instance(Estimator)
    fit_args = _make_args(estimator, "fit")
    assert (
        estimator.fit(*fit_args) is estimator
    ), f"Estimator: {estimator} does not return self when calling fit"


def check_raises_not_fitted_error(Estimator):
    # Check that we raise appropriate error for unfitted estimators
    estimator = _construct_instance(Estimator)

    # call methods without prior fitting and check that they raise our
    # NotFittedError
    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            args = _make_args(estimator, method)
            with pytest.raises(NotFittedError, match=r"has not been fitted"):
                getattr(estimator, method)(*args)


def check_fit_idempotent(Estimator):
    # Check that calling fit twice is equivalent to calling it once
    estimator = _construct_instance(Estimator)
    set_random_state(estimator)

    # Fit for the first time
    fit_args = _make_args(estimator, "fit")
    estimator.fit(*fit_args)

    results = dict()
    args = dict()
    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            args[method] = _make_args(estimator, method)
            results[method] = getattr(estimator, method)(*args[method])

    # Fit again
    set_random_state(estimator)
    estimator.fit(*fit_args)

    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(*args[method])
            _assert_array_almost_equal(
                results[method],
                new_result,
                # err_msg=f"Idempotency check failed for method {method}",
            )


def check_fit_does_not_overwrite_hyper_params(Estimator):
    # Check that we do not overwrite hyper-parameters in fit
    estimator = _construct_instance(Estimator)
    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    fit_args = _make_args(estimator, "fit")
    estimator.fit(*fit_args)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (estimator.__class__.__name__, param_name, original_value, new_value)
        )


def check_methods_do_not_change_state(Estimator):
    # Check that methods that are not supposed to change attributes of the
    # estimators do not change anything (including hyper-parameters and
    # fitted parameters)
    estimator = _construct_instance(Estimator)
    set_random_state(estimator)

    fit_args = _make_args(estimator, "fit")
    estimator.fit(*fit_args)
    dict_before = estimator.__dict__.copy()

    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            args = _make_args(estimator, method)
            getattr(estimator, method)(*args)

            if (
                is_non_fittable_series_as_features_transformer(estimator)
                and method == "transform"
            ):
                # these transformers fit during transform, as they apply
                # some function or wrapped transformer to each series,
                # so transform will actually change the state of the estimator
                continue

            assert (
                estimator.__dict__ == dict_before
            ), f"Estimator: {estimator} changes __dict__ during {method}"


def check_persistence_via_pickle(Estimator):
    # Check that we can pickle all estimators
    estimator = _construct_instance(Estimator)
    set_random_state(estimator)
    fit_args = _make_args(estimator, "fit")
    estimator.fit(*fit_args)

    # Generate results before pickling
    results = dict()
    args = dict()
    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            args[method] = _make_args(estimator, method)
            results[method] = getattr(estimator, method)(*args[method])

    # Pickle and unpickle
    pickled_estimator = pickle.dumps(estimator)
    unpickled_estimator = pickle.loads(pickled_estimator)

    # Compare against results after pickling
    for method in results:
        unpickled_result = getattr(unpickled_estimator, method)(*args[method])
        _assert_array_almost_equal(
            results[method],
            unpickled_result,
            decimal=6,
            err_msg="Results are not the same after pickling",
        )


def check_multiprocessing_idempotent(Estimator):
    # Check that running an estimator on a single process is no different to running
    # it on multiple processes. We also check that we can set n_jobs=-1 to make use
    # of all CPUs. The test is not really necessary though, as we rely on joblib for
    # parallelization and can trust that it works as expected.
    estimator = _construct_instance(Estimator)
    params = estimator.get_params()

    if "n_jobs" in params:
        results = dict()
        args = dict()

        # run on a single process
        estimator = _construct_instance(Estimator)
        estimator.set_params(n_jobs=1)
        set_random_state(estimator)
        args["fit"] = _make_args(estimator, "fit")
        estimator.fit(*args["fit"])

        # compute and store results
        for method in NON_STATE_CHANGING_METHODS:
            if hasattr(estimator, method):
                args[method] = _make_args(estimator, method)
                results[method] = getattr(estimator, method)(*args[method])

        # run on multiple processes, reusing the same input arguments
        estimator = _construct_instance(Estimator)
        estimator.set_params(n_jobs=-1)
        set_random_state(estimator)
        estimator.fit(*args["fit"])

        # compute and compare results
        for method in results:
            if hasattr(estimator, method):
                result = getattr(estimator, method)(*args[method])
                _assert_array_equal(
                    results[method],
                    result,
                    err_msg="Results are not equal for n_jobs=1 and " "n_jobs=-1",
                )
