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
# through ``transformers.AutoModelForCausalLM`` with ``trust_remote_code=True``
# and bypassing sktime entirely.
#
# ``TimerForecaster`` has no ``revision`` parameter, so the tests below load
# the ``main`` revision of the checkpoint, which at the time of writing is the
# commit above; the model code itself comes from the vendored copy in
# ``sktime.libs.timer``, which is functionally identical to that commit.
#
# Environment: Python 3.12.14, torch 2.14.0, transformers 4.40.1, numpy 1.26.4.
#
# The reference values were produced by the following standalone script:
#
#     import numpy as np
#     import torch
#     from transformers import AutoModelForCausalLM
#
#     # sktime is only used as a data source here (airline passengers).
#     from sktime.datasets import load_airline
#
#     REPO = "thuml/timer-base-84m"
#     REVISION = "70077a71acce1b4c00d98332fcaabc694255d8e5"
#     SEED = 0
#     SYNTHETIC_LENGTH = 300
#
#     # The pretrained checkpoint at REVISION (84M parameters, 337 MB of float32
#     # weights) is loaded on CPU, the torch default device. The model is
#     # deterministic: it has no dropout and generation is greedy, so SEED only
#     # pins the RNG state for good measure.
#     #
#     # Each case mirrors one TimerForecaster configuration:
#     #
#     # * "default": TimerForecaster() on the first 132 airline values, fh=1..12.
#     #   The context is the last 96 values (one input token, the remainder is
#     #   dropped), one generation step is truncated to 12 values.
#     # * "default-two-steps": as "default" with fh=1..100, so the 96 values of the
#     #   first generation step are followed by a second step through the KV cache
#     #   that yields the remaining 4 values.
#     # * "context-length-96": TimerForecaster(context_length=96) on a synthetic
#     #   seasonal series with trend of 300 values, fh=1..12. The context is cut to
#     #   the last 96 values before token rounding, i.e. one token instead of the
#     #   three tokens (288 values) that the default context length would use.
#     CASES = {
#         "default": {"series": "airline", "context_length": 2880, "horizon": 12},
#         "default-two-steps": {
#             "series": "airline",
#             "context_length": 2880,
#             "horizon": 100,
#         },
#         "context-length-96": {
#             "series": "synthetic",
#             "context_length": 96,
#             "horizon": 12,
#         },
#     }
#
#
#     def seasonal_trend_series(n):
#         t = np.arange(n, dtype=np.float64)
#         return 100.0 + 0.5 * t + 20.0 * np.sin(2 * np.pi * t / 12)
#
#
#     SERIES = {
#         "airline": lambda: load_airline().to_numpy()[:-12],
#         "synthetic": lambda: seasonal_trend_series(SYNTHETIC_LENGTH),
#     }
#
#
#     def fmt(values):
#         return "[" + ", ".join(f"{float(v):.9g}" for v in values) + "]"
#
#
#     def main():
#         model = AutoModelForCausalLM.from_pretrained(
#             REPO, revision=REVISION, trust_remote_code=True, torch_dtype=torch.float32
#         )
#         model.eval()
#
#         for name, case in CASES.items():
#             y = SERIES[case["series"]]().astype(np.float32)
#             # TimerForecaster keeps the most recent ``context_length`` values;
#             # upstream ``generate`` then drops the oldest remainder that does not
#             # fill a whole input token of config.input_token_len values.
#             context = y[-case["context_length"] :]
#             past_values = torch.tensor(context, dtype=torch.float32)[None]
#             torch.manual_seed(SEED)
#             with torch.no_grad():
#                 output = model.generate(past_values, max_new_tokens=case["horizon"])
#             # output: [batch=1, horizon]
#             forecast = output[0].numpy()
#             assert forecast.shape == (case["horizon"],), forecast.shape
#             print(f"== {name} (context {len(context)} values)")
#             print("  head :", fmt(forecast[:3]))
#             print("  tail :", fmt(forecast[-3:]))
#
#
#     if __name__ == "__main__":
#         main()

_TIMER_MODEL_NAME = "thuml/timer-base-84m"
_TIMER_SYNTHETIC_LENGTH = 300


def _seasonal_trend_series(n):
    """Deterministic monthly series with a linear trend and a yearly season."""
    t = np.arange(n, dtype=np.float64)
    values = 100.0 + 0.5 * t + 20.0 * np.sin(2 * np.pi * t / 12)
    index = pd.period_range("2000-01", periods=n, freq="M")
    return pd.Series(values, index=index, name="synthetic")


def _load_airline_train():
    """Airline passengers without the last 12 months."""
    return load_airline().iloc[:-12]


# (series loader, context_length, horizon, expected head, expected tail)
_TIMER_UPSTREAM_REFERENCE_CASES = [
    pytest.param(
        _load_airline_train,
        2880,
        12,
        [393.205109, 401.126251, 405.195679],
        [377.61911, 373.585724, 385.950806],
        id="default",
    ),
    pytest.param(
        _load_airline_train,
        2880,
        100,
        [393.205109, 401.126251, 405.195679],
        [343.705017, 345.483154, 347.02417],
        id="default-two-steps",
    ),
    pytest.param(
        lambda: _seasonal_trend_series(_TIMER_SYNTHETIC_LENGTH),
        96,
        12,
        [246.726791, 256.788849, 262.056763],
        [220.53685, 223.281296, 231.372559],
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
        model_name=_TIMER_MODEL_NAME,
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
