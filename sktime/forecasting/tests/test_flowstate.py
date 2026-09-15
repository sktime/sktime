# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for FlowStateForecaster probabilistic forecasts."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.flowstate import FlowStateForecaster

pytestmark = pytest.mark.skipif(
    not _check_estimator_deps(FlowStateForecaster, severity="none"),
    reason="missing deps for Flowstate tests",
)


@pytest.mark.parametrize(
    "alpha",
    [
        [0.1, 0.5, 0.9],
        [0.3, 0.7],
        [0.01, 0.99],
    ],
)
def test_flowstate_predict_proba_histogram_consistency(alpha):
    """FlowState predict_proba returns HistogramQPD with consistent quantiles."""
    from skpro.distributions import HistogramQPD

    params = FlowStateForecaster.get_test_params()[0]
    forecaster = FlowStateForecaster(**params)
    index = pd.RangeIndex(10, name="time")
    y = pd.DataFrame({"y": np.arange(10, dtype=float)}, index=index)
    fh = [1, 3]
    forecaster.fit(y, fh=fh)

    pred_dist = forecaster.predict_proba(fh=fh)
    assert isinstance(pred_dist, HistogramQPD)

    quantiles_from_predict = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    quantiles_from_proba = pred_dist.quantile(alpha=alpha)
    pd.testing.assert_index_equal(
        quantiles_from_predict.index, quantiles_from_proba.index
    )
    np.testing.assert_allclose(quantiles_from_predict, quantiles_from_proba)
