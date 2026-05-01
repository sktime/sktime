"""Tests for conditional transforms using TransformIf."""

import pytest

from sktime.param_est.seasonality import SeasonalityACF
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose import Id, TransformIf
from sktime.transformations.series.detrend import Deseasonalizer


@pytest.mark.skipif(
    not run_test_for_class([SeasonalityACF, Deseasonalizer, TransformIf]),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
def test_conditional_deseasonalization():
    """Test deaseaonalizer TransformIf, same as docstring."""
    # pipeline with deseasonalization conditional on seasonality test
    from sktime.datasets import load_airline

    y = load_airline()

    seasonal = SeasonalityACF(candidate_sp=12)
    deseason = Deseasonalizer(sp=12)
    cond_deseason = TransformIf(seasonal, "sp", "!=", 1, deseason)
    y_hat = cond_deseason.fit_transform(y)

    assert len(y_hat) == len(y)
    assert (y_hat.index == y.index).all()


class _DummyFittedEstimator:
    """Dummy estimator with fitted params for condition evaluation tests."""

    def __init__(self, params):
        self._params = params

    def get_fitted_params(self):
        return self._params


@pytest.mark.skipif(
    not run_test_for_class([TransformIf]),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "condition,condition_value,expected",
    [
        ("bool", None, "if"),
        (">=", 3, "if"),
        (">", 2, "if"),
        ("==", 3, "if"),
        ("!=", 2, "if"),
        ("<", 4, "if"),
        ("<=", 3, "if"),
        (">", 4, "else"),
    ],
)
def test_transformif_evaluate_condition(condition, condition_value, expected):
    """Test condition evaluation branches in TransformIf."""
    trafo = TransformIf(if_estimator=Id(), param="score")
    trafo.if_estimator_ = _DummyFittedEstimator({"score": 3, "flag": True})
    trafo.condition = condition
    trafo.condition_value = condition_value

    assert trafo._evaluate_condition() == expected
