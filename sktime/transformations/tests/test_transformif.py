"""Tests for conditional transforms using TransformIf."""

import pytest

from sktime.param_est.seasonality import SeasonalityACF
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose import TransformIf
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
