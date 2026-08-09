# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for SundialForecaster probabilistic API."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.sundial import SundialForecaster

pytestmark = pytest.mark.skipif(
    not _check_estimator_deps(SundialForecaster, severity="none"),
    reason="torch, transformers, and skpro required for Sundial probabilistic tests",
)


@pytest.mark.parametrize(
    "alpha",
    [
        [0.1, 0.5, 0.9],
        [0.25, 0.75],
        [0.001, 0.999],
    ],
)
def test_sundial_predict_proba_empirical_consistency(alpha):
    """Sundial predict_proba returns Empirical with consistent quantiles."""
    from skpro.distributions.empirical import Empirical

    params = SundialForecaster.get_test_params()[0]
    forecaster = SundialForecaster(**params)
    y = pd.DataFrame({"y": np.arange(10, dtype=float)})
    fh = [1, 2, 3]

    forecaster.fit(y, fh=fh)

    pred_dist = forecaster.predict_proba(fh=fh)
    assert isinstance(pred_dist, Empirical)

    quantiles_from_predict = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    quantiles_from_proba = pred_dist.quantile(alpha=alpha)
    np.testing.assert_allclose(quantiles_from_predict, quantiles_from_proba)
