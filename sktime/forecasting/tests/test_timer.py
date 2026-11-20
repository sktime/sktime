# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for the Timer forecaster.

The reference outputs in this module are generated from the pretrained
``thuml/timer-base-84m`` checkpoint (84M parameters, 337 MB in float32),
which fits comfortably in memory, so no seeded-small-model fallback is used.

The checkpoint's remote code, and hence ``TimerForecaster``, only supports
``transformers`` 4.40.x: newer releases removed the cache and generation
internals it relies on. The references were therefore generated, and the
tests below run, with transformers 4.40.1, the version pinned by the upstream
model card.
"""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timer import TimerForecaster
from sktime.tests.test_switch import run_test_for_class

# Reference forecasts were generated on CPU from the upstream Timer remote code
# in the thuml/timer-base-84m Hugging Face repository at the immutable commit
# 70077a71acce1b4c00d98332fcaabc694255d8e5 (model code and weights), loaded
# through ``transformers.AutoModelForCausalLM`` with ``trust_remote_code=True``,
# in a venv without sktime. The airline values are embedded in the script.
#
# ``TimerForecaster`` has no ``revision`` parameter, so the tests below
# download the checkpoint at the commit above with ``snapshot_download`` and
# pass the local path as ``model_name``. This also avoids parallel workers
# racing on the repo's ``refs/main`` under ``pytest -n auto``. The model code
# comes from the vendored copy in ``sktime.libs.timer``, which is functionally
# identical to that commit.
#
# Environment: Python 3.12.13, torch 2.15.0 (CPU), transformers 4.40.1,
# numpy 1.26.4. Two runs gave identical values; the model has no dropout and
# generation is greedy.
#
# The reference values were produced by the following standalone script:
#
#     import numpy as np
#     import torch
#     from transformers import AutoModelForCausalLM
#
#     REPO = "thuml/timer-base-84m"
#     REVISION = "70077a71acce1b4c00d98332fcaabc694255d8e5"
#     INPUT_TOKEN_LEN = 96
#
#     # values of sktime.datasets.load_airline, monthly from 1949-01
#     AIRLINE_VALUES = [
#         112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
#         115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
#         145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
#         171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
#         196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
#         204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
#         242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
#         284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
#         315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
#         340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
#         360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
#         417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
#     ]
#
#
#     def seasonal_trend(n=300):
#         t = np.arange(n, dtype=np.float64)
#         return 100.0 + 0.5 * t + 20.0 * np.sin(2 * np.pi * t / 12)
#
#
#     # name: (series, context_length, horizon), horizon = max(fh)
#     CASES = {
#         # context 2880 > 132 values, so 96 values (one token) are used
#         "default": (np.asarray(AIRLINE_VALUES[:-12]), 2880, 12),
#         # 100 > output_token_len 96: second generation step via the KV cache
#         "default-two-steps": (np.asarray(AIRLINE_VALUES[:-12]), 2880, 100),
#         # context_length cuts 300 values to 96 instead of 288 (three tokens)
#         "context-length-96": (seasonal_trend(), 96, 12),
#     }
#
#
#     def build_context(y, context_length):
#         # mirrors TimerForecaster: float32, zero padding at the front up to
#         # one token, last context_length values, whole tokens only
#         y = y.astype(np.float32)
#         if len(y) < INPUT_TOKEN_LEN:
#             pad = np.zeros(INPUT_TOKEN_LEN - len(y), np.float32)
#             y = np.concatenate([pad, y])
#         y = y[-context_length:]
#         return y[-(len(y) // INPUT_TOKEN_LEN) * INPUT_TOKEN_LEN :]
#
#
#     model = AutoModelForCausalLM.from_pretrained(
#         REPO, revision=REVISION, trust_remote_code=True, torch_dtype=torch.float32
#     )
#     model.eval()
#     for name, (y, context_length, horizon) in CASES.items():
#         past = torch.tensor(build_context(y, context_length))[None]
#         with torch.no_grad():
#             forecast = model.generate(past, max_new_tokens=horizon)[0].numpy()
#         assert forecast.shape == (horizon,)
#         print(name, forecast[:3].tolist(), forecast[-3:].tolist())

_TIMER_REPO = "thuml/timer-base-84m"
_TIMER_REVISION = "70077a71acce1b4c00d98332fcaabc694255d8e5"
_TIMER_SYNTHETIC_LENGTH = 300


def _seasonal_trend_series(n):
    """Deterministic monthly series with a linear trend and a yearly season."""
    t = np.arange(n, dtype=np.float64)
    values = 100.0 + 0.5 * t + 20.0 * np.sin(2 * np.pi * t / 12)
    index = pd.period_range("2000-01", periods=n, freq="M")
    return pd.Series(values, index=index, name="synthetic")


def _pinned_checkpoint():
    """Local path of the checkpoint at the pinned revision."""
    from huggingface_hub import snapshot_download

    return snapshot_download(_TIMER_REPO, revision=_TIMER_REVISION)


def _load_airline_train():
    """Airline passengers without the last 12 months."""
    return load_airline().iloc[:-12]


# (series loader, context_length, horizon, expected head, expected tail)
_TIMER_UPSTREAM_REFERENCE_CASES = [
    pytest.param(
        _load_airline_train,
        2880,
        12,
        [393.205139, 401.126282, 405.195709],
        [377.61911, 373.585754, 385.950836],
        id="default",
    ),
    pytest.param(
        _load_airline_train,
        2880,
        100,
        [393.205139, 401.126282, 405.195709],
        [343.705017, 345.483154, 347.02417],
        id="default-two-steps",
    ),
    pytest.param(
        lambda: _seasonal_trend_series(_TIMER_SYNTHETIC_LENGTH),
        96,
        12,
        [246.726776, 256.788849, 262.056763],
        [220.53685, 223.281311, 231.372559],
        id="context-length-96",
    ),
]


pytestmark = pytest.mark.skipif(
    not run_test_for_class(TimerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


@pytest.mark.parametrize(
    "load_series,context_length,horizon,expected_head,expected_tail",
    _TIMER_UPSTREAM_REFERENCE_CASES,
)
def test_timer_predictions_match_upstream_reference(
    load_series, context_length, horizon, expected_head, expected_tail
):
    """Timer predictions match the upstream thuml/timer-base-84m model outputs."""
    y_train = load_series()
    fh = np.arange(1, horizon + 1)
    forecaster = TimerForecaster(
        model_name=_pinned_checkpoint(),
        context_length=context_length,
        device="cpu",
    )

    y_pred = forecaster.fit(y_train, fh=fh).predict(fh=fh)

    assert len(y_pred) == horizon
    np.testing.assert_allclose(
        y_pred.iloc[:3].to_numpy(),
        np.asarray(expected_head, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        y_pred.iloc[-3:].to_numpy(),
        np.asarray(expected_tail, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )
