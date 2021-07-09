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
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn import clone
from sklearn.utils.estimator_checks import (
    check_get_params_invariance as _check_get_params_invariance,
)
from sklearn.utils.estimator_checks import check_set_params as _check_set_params
from sklearn.utils._testing import set_random_state
from sklearn.utils.validation import check_random_state

from sktime.base import BaseEstimator
from sktime.classification.base import BaseClassifier
from sktime.exceptions import NotFittedError
from sktime.forecasting.base import BaseForecaster
from sktime.regression.base import BaseRegressor
from sktime.tests._config import ESTIMATOR_TEST_PARAMS
from sktime.tests._config import NON_STATE_CHANGING_METHODS
from sktime.tests._config import VALID_ESTIMATOR_BASE_TYPES
from sktime.tests._config import VALID_ESTIMATOR_TYPES
from sktime.tests._config import VALID_TRANSFORMER_TYPES
from sktime.tests._config import VALID_ESTIMATOR_TAGS
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.base import _PanelToTabularTransformer
from sktime.transformations.base import _SeriesToPrimitivesTransformer
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils._testing.forecasting import _make_series
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.panel import _make_panel_X
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils._testing.panel import make_regression_problem
from sktime.utils._testing.panel import make_clustering_problem
from sktime.utils.data_processing import is_nested_dataframe
from sktime.utils import _has_tag
from sktime.clustering.base.base import BaseClusterer

from sktime.annotation.base import BaseSeriesAnnotator
from sktime.utils._testing.annotation import make_annotation_problem


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
        check_estimator_tags,
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
        check_valid_estimator_tags,
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
            f"tuple, but found type: {type(required_params)}"
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


def check_estimator_tags(Estimator):
    assert hasattr(Estimator, "_all_tags")
    all_tags = Estimator._all_tags()
    assert isinstance(all_tags, dict)
    assert all([isinstance(key, str) for key in all_tags.keys()])
    if hasattr(Estimator, "_tags"):
        tags = Estimator._tags
        assert isinstance(tags, dict), f"_tags must be a dict, but found {type(tags)}"
        assert len(tags) > 0, "_tags is empty"
        assert all(
            [tag in VALID_ESTIMATOR_TAGS for tag in tags.keys()]
        ), "Some tags in _tags are invalid"

    # Avoid ambiguous class attributes
    ambiguous_attrs = ("tags", "tags_")
    for attr in ambiguous_attrs:
        assert not hasattr(Estimator, attr), (
            f"Please avoid using the {attr} attribute to disambiguate it from "
            f"estimator tags."
        )


def check_inheritance(Estimator):
    # Check that estimator inherits from BaseEstimator
    assert issubclass(Estimator, BaseEstimator), (
        f"Estimator: {Estimator} " f"is not a sub-class of " f"BaseEstimator."
    )

    # Usually estimators inherit only from one BaseEstimator type, but in some cases
    # they may be predictor and transformer at the same time (e.g. pipelines)
    n_base_types = sum(
        [issubclass(Estimator, cls) for cls in VALID_ESTIMATOR_BASE_TYPES]
    )
    assert 2 >= n_base_types >= 1

    # If the estimator inherits from more than one base estimator type, we check if
    # one of them is a transformer base type
    if n_base_types > 1:
        assert issubclass(Estimator, VALID_TRANSFORMER_TYPES)


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
    if hasattr(estimator, "inverse_transform"):
        assert hasattr(estimator, "transform")
    if hasattr(estimator, "predict_proba"):
        assert hasattr(estimator, "predict")


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
    required_params = getattr(estimator, "_required_parameters", tuple())
    test_params = ESTIMATOR_TEST_PARAMS.get(Estimator, {}).keys()

    init_params = [
        param
        for param in init_params
        if param.name not in required_params and param.name not in test_params
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

            if method == "transform" and _has_tag(Estimator, "fit-in-transform"):
                # Some transformations fit during transform, as they apply
                # some transformation to each series passed to transform,
                # so transform will actually change the state of these estimator.
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
                    err_msg="Results are not equal for n_jobs=1 and n_jobs=-1",
                )


def check_valid_estimator_tags(Estimator):
    # check if Estimator tags are in VALID_ESTIMATOR_TAGS
    for tag in Estimator._all_tags().keys():
        assert tag in VALID_ESTIMATOR_TAGS


def _get_err_msg(estimator):
    return (
        f"Invalid estimator type: {type(estimator)}. Valid estimator types are: "
        f"{VALID_ESTIMATOR_TYPES}"
    )


def _construct_instance(Estimator):
    """Construct Estimator instance if possible"""
    # if non-default parameters are required, but none have been found,
    # raise error
    if hasattr(Estimator, "_required_parameters"):
        required_parameters = getattr(Estimator, "required_parameters", [])
        if len(required_parameters) > 0:
            raise ValueError(
                f"Estimator: {Estimator} requires "
                f"non-default parameters for construction, "
                f"but none were given. Please set them in "
                f"sktime/tests/_config.py"
            )

    # construct with parameter configuration for testing, otherwise construct with
    # default parameters (empty param dict)
    params = ESTIMATOR_TEST_PARAMS.get(Estimator, {})
    return Estimator(**params)


def _make_args(estimator, method, **kwargs):
    """Helper function to generate appropriate arguments for testing different
    estimator types and their methods"""
    if method == "fit":
        return _make_fit_args(estimator, **kwargs)
    if method == "update":
        raise NotImplementedError()
    elif method in ("predict", "predict_proba", "decision_function"):
        return _make_predict_args(estimator, **kwargs)
    elif method == "transform":
        return _make_transform_args(estimator, **kwargs)
    elif method == "inverse_transform":
        return _make_inverse_transform_args(estimator, **kwargs)
    else:
        raise ValueError(f"Method: {method} not supported")


def _make_fit_args(estimator, **kwargs):
    if isinstance(estimator, BaseForecaster):
        # we need to handle the TransformedTargetForecaster separately
        if isinstance(estimator, _SeriesToSeriesTransformer):
            y = _make_series(**kwargs)
        else:
            y = make_forecasting_problem(**kwargs)
        fh = 1
        X = None
        return y, X, fh
    elif isinstance(estimator, BaseSeriesAnnotator):
        X = make_annotation_problem(**kwargs)
        return (X,)
    elif isinstance(estimator, BaseClassifier):
        return make_classification_problem(**kwargs)
    elif isinstance(estimator, BaseRegressor):
        return make_regression_problem(**kwargs)
    elif isinstance(
        estimator, (_SeriesToPrimitivesTransformer, _SeriesToSeriesTransformer)
    ):
        X = _make_series(**kwargs)
        return (X,)
    elif isinstance(estimator, (_PanelToTabularTransformer, _PanelToPanelTransformer)):
        return make_classification_problem(**kwargs)
    elif isinstance(estimator, BaseClusterer):
        return (make_clustering_problem(**kwargs),)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_predict_args(estimator, **kwargs):
    if isinstance(estimator, BaseForecaster):
        fh = 1
        return (fh,)
    elif isinstance(estimator, (BaseClassifier, BaseRegressor)):
        X = _make_panel_X(**kwargs)
        return (X,)
    elif isinstance(estimator, BaseSeriesAnnotator):
        X = make_annotation_problem(n_timepoints=10, **kwargs)
        return (X,)
    elif isinstance(estimator, BaseClusterer):
        X = _make_panel_X(**kwargs)
        return (X,)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_transform_args(estimator, **kwargs):
    if isinstance(
        estimator, (_SeriesToPrimitivesTransformer, _SeriesToSeriesTransformer)
    ):
        X = _make_series(**kwargs)
        return (X,)
    elif isinstance(
        estimator,
        (
            _PanelToTabularTransformer,
            _PanelToPanelTransformer,
        ),
    ):
        X = _make_panel_X(**kwargs)
        return (X,)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_inverse_transform_args(estimator, **kwargs):
    if isinstance(estimator, _SeriesToPrimitivesTransformer):
        X = _make_primitives(**kwargs)
        return (X,)
    elif isinstance(estimator, _SeriesToSeriesTransformer):
        X = _make_series(**kwargs)
        return (X,)
    elif isinstance(estimator, _PanelToTabularTransformer):
        X = _make_tabular_X(**kwargs)
        return (X,)
    elif isinstance(estimator, _PanelToPanelTransformer):
        X = _make_panel_X(**kwargs)
        return (X,)
    else:
        raise ValueError(_get_err_msg(estimator))


def _make_primitives(n_columns=1, random_state=None):
    """Generate one or more primitives. Useful for checking inverse-transform
    of series-to-primitives transformer"""
    rng = check_random_state(random_state)
    if n_columns == 1:
        return rng.rand()
    return rng.rand(size=(n_columns,))


def _make_tabular_X(n_instances=20, n_columns=1, return_numpy=True, random_state=None):
    """Generate tabular X. Useful for checking inverse-transform
    of panel-to-tabular transformer"""
    rng = check_random_state(random_state)
    X = rng.rand(n_instances, n_columns)
    if return_numpy:
        return X
    else:
        return pd.DataFrame(X)


def _compare_nested_frame(func, x, y, **kwargs):
    """Helper function to compare two nested pd.DataFrames

    Parameters
    ----------
    func : function
        Function from np.testing for comparing arrays.
    x : pd.DataFrame
    y : pd.DataFrame
    kwargs : dict
        Keyword argument for function

    Raises
    ------
    AssertionError
        If x and y are not equal
    """
    # We iterate over columns and rows to make cell-wise comparisons.
    # Tabularizing the data first would simplify this, but does not
    # work for unequal length data.

    # In rare cases, x and y may be empty (e.g. TSFreshRelevantFeatureExtractor) and
    # we cannot compare individual cells, so we simply check if everything else is
    # equal here.
    assert isinstance(x, pd.DataFrame)
    if x.empty:
        assert_frame_equal(x, y)

    elif is_nested_dataframe(x):
        # Check if both inputs have the same shape
        if not x.shape == y.shape:
            raise ValueError("Found inputs with different shapes")

        # Iterate over columns
        n_columns = x.shape[1]
        for i in range(n_columns):
            xc = x.iloc[:, i].tolist()
            yc = y.iloc[:, i].tolist()

            # Iterate over rows, checking if individual cells are equal
            for xci, yci in zip(xc, yc):
                func(xci, yci, **kwargs)


def _assert_array_almost_equal(x, y, decimal=6, err_msg=""):
    func = np.testing.assert_array_almost_equal
    if isinstance(x, pd.DataFrame):
        _compare_nested_frame(func, x, y, decimal=decimal, err_msg=err_msg)
    else:
        func(x, y, decimal=decimal, err_msg=err_msg)


def _assert_array_equal(x, y, err_msg=""):
    func = np.testing.assert_array_equal
    if isinstance(x, pd.DataFrame):
        _compare_nested_frame(func, x, y, err_msg=err_msg)
    else:
        func(x, y, err_msg=err_msg)


def _get_args(function, varargs=False):
    """Helper to get function arguments"""
    try:
        params = signature(function).parameters
    except ValueError:
        # Error on builtin C function
        return []
    args = [
        key
        for key, param in params.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    if varargs:
        varargs = [
            param.name
            for param in params.values()
            if param.kind == param.VAR_POSITIONAL
        ]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args
