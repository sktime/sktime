# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for DieboldMarianoTest."""

import numpy as np
import pandas as pd
import pytest

from sktime.benchmarking.analysis import DieboldMarianoTest
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency statsmodels not available",
)
@pytest.mark.parametrize("harvey_adj", [True, False])
def test_diebold_mariano_parity_with_statsmodels(harvey_adj):
    """Test that DieboldMarianoTest matches statsmodels reference output."""
    from statsmodels.tsa.stattools import diebold_mariano_test as sm_dm_test

    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=60)
    pred_a = y + rng.normal(0.0, 0.2, size=60)
    pred_b = y + rng.normal(0.0, 1.0, size=60)

    analyzer = DieboldMarianoTest(horizon=2, harvey_adj=harvey_adj)
    res = analyzer.evaluate(pd.Series(y), pd.Series(pred_a), pd.Series(pred_b))
    ref = sm_dm_test(y, pred_a, pred_b, horizon=2, harvey_adj=harvey_adj)

    assert isinstance(res, pd.DataFrame)
    assert np.isclose(res["statistic"].iloc[0], ref.statistic)
    assert np.isclose(res["p_val"].iloc[0], ref.pvalue)
    assert res["lags"].iloc[0] == ref.lags
    assert res["nobs"].iloc[0] == 60
    assert res["statistic"].iloc[0] < 0
    assert res["p_val"].iloc[0] < 0.05


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency statsmodels not available",
)
def test_diebold_mariano_validation_errors():
    """Test input validation for length mismatch, horizon, and min observations."""
    y = pd.Series([1.0, 2.0, 3.0])
    fa = pd.Series([1.1, 2.1, 3.1])
    fb = pd.Series([1.2, 2.2])

    analyzer = DieboldMarianoTest()
    with pytest.raises(ValueError, match="same length"):
        analyzer.evaluate(y, fa, fb)

    with pytest.raises(ValueError, match="horizon"):
        DieboldMarianoTest(horizon=0).evaluate(y, fa, fa)

    with pytest.raises(ValueError, match="At least 2 observations"):
        analyzer.evaluate([1.0], [1.1], [1.2])
