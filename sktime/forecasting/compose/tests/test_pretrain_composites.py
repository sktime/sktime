"""Tests for state-aware set_params in composite forecasters."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["kr1xh"]

from copy import deepcopy

import pytest

from sktime.forecasting.compose import (
    EnsembleForecaster,
    MultiplexForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.exponent import ExponentTransformer
from sktime.utils._testing.hierarchical import _make_hierarchical


def _make_pretrained_dummy():
    """Return a DummyGlobalForecaster pretrained on a small panel."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(3,),
        min_timepoints=10,
        max_timepoints=10,
        n_columns=1,
        random_state=0,
    )
    forecaster = DummyGlobalForecaster()
    forecaster.pretrain(y_panel)
    return forecaster


def test_nested_set_params_at_keeps_pretrained_state():
    """Dotted params reach a pretrained component without wiping its state."""
    forecaster = _make_pretrained_dummy()
    pretrain_mean = forecaster.global_mean_
    pipe = TransformedTargetForecaster(
        [("exp", ExponentTransformer(power=1)), ("forecaster", forecaster)]
    )

    pipe._set_params_at("pretrained", {"forecaster__strategy": "last", "exp__power": 2})

    inner = pipe.named_steps["forecaster"]
    assert inner.strategy == "last"
    assert inner.global_mean_ == pytest.approx(pretrain_mean)
    # component without _set_params_at falls back to plain set_params
    assert pipe.named_steps["exp"].power == 2


def test_step_replacement_via_set_params_at():
    """Replacing a step by name must work in the state-aware path."""
    pipe = TransformedTargetForecaster(
        [
            ("exp", ExponentTransformer(power=1)),
            ("forecaster", _make_pretrained_dummy()),
        ]
    )
    new = DummyGlobalForecaster(strategy="last")

    pipe._set_params_at("pretrained", {"forecaster": new})

    assert pipe.named_steps["forecaster"] is new


def _make_ttf():
    return TransformedTargetForecaster(
        [("exp", ExponentTransformer(power=1)), ("forecaster", NaiveForecaster())]
    )


def _make_ensemble():
    return EnsembleForecaster(
        [("a", NaiveForecaster()), ("b", NaiveForecaster(strategy="drift"))]
    )


def _make_multiplex():
    return MultiplexForecaster(
        [("a", NaiveForecaster()), ("b", NaiveForecaster(strategy="drift"))]
    )


PARITY_CASES = [
    (_make_ttf, {"exp__power": 2, "forecaster__strategy": "mean"}),
    (_make_ttf, {"forecaster": NaiveForecaster(strategy="drift")}),
    (_make_ensemble, {"a__strategy": "mean"}),
    (_make_ensemble, {"a": NaiveForecaster(strategy="mean")}),
    (_make_multiplex, {"selected_forecaster": "b"}),
]


def _param_reprs(estimator):
    return {key: repr(value) for key, value in estimator.get_params().items()}


@pytest.mark.parametrize("make, params", PARITY_CASES)
def test_set_params_at_new_equals_set_params(make, params):
    """_set_params_at('new', p) must behave exactly like set_params(**p)."""
    expected = make().set_params(**deepcopy(params))
    actual = make()._set_params_at("new", deepcopy(params))

    # compare deep params, since repr omits replaced steps of some composites
    assert _param_reprs(actual) == _param_reprs(expected)
