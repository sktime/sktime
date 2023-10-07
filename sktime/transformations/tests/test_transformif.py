"""Tests for conditional transforms using TransformIf."""

import pytest

from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_conditional_deseasonalization():
    """Test deaseaonalizer TransformIf, same as docstring."""
    # pipeline with deseasonalization conditional on seasonality test
    from sktime.datasets import load_airline
    from sktime.param_est.seasonality import SeasonalityACF
    from sktime.transformations.compose import TransformIf
    from sktime.transformations.series.detrend import Deseasonalizer

    y = load_airline()

    seasonal = SeasonalityACF(candidate_sp=12)
    deseason = Deseasonalizer(sp=12)
    cond_deseason = TransformIf(seasonal, "sp", "!=", 1, deseason)
    y_hat = cond_deseason.fit_transform(y)

    assert len(y_hat) == len(y)
    assert (y_hat.index == y.index).all()
