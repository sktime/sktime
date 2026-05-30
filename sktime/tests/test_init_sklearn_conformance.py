#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for sklearn ``__init__`` parameter storage conformance.

These tests guard against the class of bugs reported in #10208, where some
estimators either used mutable container literals as default arguments, or
mutated/derived ``__init__`` parameters and stored them on attributes of the
same name. Both break sklearn's instantiation contract:
https://scikit-learn.org/stable/developers/develop.html#instantiation
"""

__author__ = ["sktime developers"]

import inspect

import pytest

from sktime.tests.test_switch import run_test_for_class


def _get_default(cls, param_name):
    """Return the default value of ``param_name`` in ``cls.__init__``."""
    return inspect.signature(cls.__init__).parameters[param_name].default


# (estimator import path, parameters that previously had mutable list defaults)
MUTABLE_DEFAULT_CASES = [
    (
        "sktime.classification.deep_learning.gru",
        "GRUFCNNClassifier",
        ["conv_layers", "kernel_sizes"],
        {"hidden_dim": 8, "gru_layers": 1},
    ),
    (
        "sktime.forecasting.rbf_forecaster",
        "RBFForecaster",
        ["hidden_layers"],
        {},
    ),
]


@pytest.mark.parametrize(
    "module, cls_name, params, extra_kwargs", MUTABLE_DEFAULT_CASES
)
def test_no_mutable_default_arguments(module, cls_name, params, extra_kwargs):
    """Container-valued parameters must default to None, not a shared literal.

    A list literal default is shared across all instances, so in-place mutation
    on one instance silently corrupts the default for every future instance.
    """
    import importlib

    cls = getattr(importlib.import_module(module), cls_name)
    if not run_test_for_class(cls):
        pytest.skip("soft dependencies not present")

    for param in params:
        default = _get_default(cls, param)
        assert default is None, (
            f"{cls_name}.__init__ parameter '{param}' has a mutable default "
            f"{default!r}; container defaults must be None and the real default "
            f"applied inside the estimator (see #10208)."
        )

    # with a None default there is no shared container literal to leak between
    # instances; confirm the stored attribute matches the (None) default
    instance = cls(**extra_kwargs)
    for param in params:
        assert getattr(instance, param) is None


MCDCNN_CASES = [
    (
        "sktime.classification.deep_learning.mcdcnn",
        "MCDCNNClassifierTorch",
    ),
    (
        "sktime.regression.deep_learning.mcdcnn._mcdcnn_torch",
        "MCDCNNRegressorTorch",
    ),
]


@pytest.mark.parametrize("module, cls_name", MCDCNN_CASES)
def test_optim_params_stored_as_is(module, cls_name):
    """``optim`` / ``optim_kwargs`` must be stored exactly as passed.

    Previously the effective optimizer defaults were derived in ``__init__``;
    the derived values must not overwrite the declared parameters, otherwise
    ``get_params()`` and ``clone()`` no longer round-trip (see #10208).
    """
    import importlib

    from sklearn.base import clone

    cls = getattr(importlib.import_module(module), cls_name)
    if not run_test_for_class(cls):
        pytest.skip("soft dependencies not present")

    est = cls(optim=None, optim_kwargs=None)
    params = est.get_params()
    assert params["optim"] is None
    assert params["optim_kwargs"] is None

    # clone round-trips to behaviourally equivalent parameters
    assert clone(est).get_params()["optim_kwargs"] is None

    # explicit user kwargs are preserved verbatim
    est2 = cls(optim="Adam", optim_kwargs={"weight_decay": 0.001})
    assert est2.get_params()["optim"] == "Adam"
    assert est2.get_params()["optim_kwargs"] == {"weight_decay": 0.001}
