# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for FlowStateForecaster probabilistic forecasts."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.datasets import load_airline
from sktime.forecasting.flowstate import FlowStateForecaster
from sktime.tests.test_switch import run_test_for_class

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


# Reference forecasts were generated on CPU from the upstream
# tsfm_public.FlowStateForPrediction, against checkpoint
# ibm-research/flowstate at revision r1.1, which resolves to the immutable
# Hugging Face commit 940580f969036cfca514cdff725f7b02c03e1bcf.
_FLOWSTATE_GRANITE_REFERENCE_CASES = [
    pytest.param(
        1.0,
        [
            399.65942,
            389.4466,
            432.6386,
        ],
        id="scale-factor-1.0-default",
    ),
    pytest.param(
        0.5,
        [
            426.69104,
            429.21277,
            460.16183,
        ],
        id="scale-factor-0.5",
    ),
]


@pytest.mark.skipif(
    not run_test_for_class(FlowStateForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "scale_factor,expected_head",
    _FLOWSTATE_GRANITE_REFERENCE_CASES,
)
def test_flowstate_airline_predictions_match_granite_reference(
    scale_factor, expected_head
):
    """FlowState predictions match the upstream granite-tsfm model outputs."""
    y = load_airline()
    y_train = y.iloc[:-12]
    fh = np.arange(1, 13)
    forecaster = FlowStateForecaster(scale_factor=scale_factor)

    y_pred = forecaster.fit(y_train, fh=fh).predict(fh=fh)

    np.testing.assert_allclose(
        y_pred.iloc[:3].to_numpy(),
        np.asarray(expected_head, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )
