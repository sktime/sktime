"""Tests for AutoResearchForecaster.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["OfficialAbhinavSingh"]

import json
import math

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.agentic import AutoResearchForecaster
from sktime.split import SingleWindowSplitter
from sktime.tests.test_switch import run_test_for_class

# ``Deseasonalizer`` with a period close to the series length leaves NaNs in the
# target, so the blueprint evaluates without raising but scores nan.
NAN_BLUEPRINT = {
    "name": "nan_blueprint",
    "spec": "Deseasonalizer(sp=59)",
    "reason": "evaluates without error but scores nan",
}
GOOD_BLUEPRINT = {
    "name": "good_blueprint",
    "spec": 'NaiveForecaster(strategy="mean")',
    "reason": "scores a finite value",
}


def _llm_returning(blueprints):
    """Build an ``llm_func`` that always returns ``blueprints``."""

    def llm_func(_messages, _model, _api_params):
        return json.dumps({"blueprints": blueprints})

    return llm_func


def _make_forecaster(blueprints):
    return AutoResearchForecaster(
        cv=SingleWindowSplitter(fh=[1, 2, 3]),
        model="dummy",
        n_iterations=1,
        n_blueprints=len(blueprints),
        llm_func=_llm_returning(blueprints),
    )


@pytest.mark.skipif(
    not run_test_for_class(AutoResearchForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "blueprints",
    [
        [NAN_BLUEPRINT, GOOD_BLUEPRINT],
        [GOOD_BLUEPRINT, NAN_BLUEPRINT],
    ],
    ids=["nan_first", "nan_second"],
)
def test_nan_score_does_not_mask_finite_blueprint(blueprints):
    """A blueprint scoring nan must not discard one that scored finitely.

    ``min`` over a list containing nan is order-dependent, and ``nan < x`` is
    always False. Before the fix, a nan-scoring blueprint listed first won the
    selection and left ``best_overall_result`` unset, so ``fit`` raised
    "No blueprint succeeded" even though another blueprint had scored finitely.
    """
    y = load_airline()[:60]
    forecaster = _make_forecaster(blueprints)

    forecaster.fit(y, fh=[1, 2, 3])

    assert forecaster.best_blueprint_["name"] == "good_blueprint"
    assert math.isfinite(forecaster.best_score_)


@pytest.mark.skipif(
    not run_test_for_class(AutoResearchForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_selection_is_order_independent():
    """The same blueprints must select the same winner in either order."""
    y = load_airline()[:60]

    scores = []
    orderings = (
        [NAN_BLUEPRINT, GOOD_BLUEPRINT],
        [GOOD_BLUEPRINT, NAN_BLUEPRINT],
    )
    for blueprints in orderings:
        forecaster = _make_forecaster(blueprints)
        forecaster.fit(y, fh=[1, 2, 3])
        scores.append(forecaster.best_score_)

    assert scores[0] == scores[1]


@pytest.mark.skipif(
    not run_test_for_class(AutoResearchForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_all_nan_blueprints_still_raise():
    """If every blueprint scores nan, fit must still report failure."""
    y = load_airline()[:60]
    forecaster = _make_forecaster([NAN_BLUEPRINT])

    with pytest.raises(RuntimeError, match="No blueprint succeeded"):
        forecaster.fit(y, fh=[1, 2, 3])
