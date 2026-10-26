# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for the TimerS1 forecaster.

The reference outputs in this module are generated from a small model with
seeded random weights, not from the pretrained checkpoint, because the only
Timer-S1 checkpoint has 8.3B parameters (about 33 GB in float32).

The tests therefore verify code-path parity, not checkpoint parity: the
vendored model in ``sktime.libs.timer_s1`` plus the ``TimerS1Forecaster``
wrapper are compared against the upstream model class from the
``bytedance-research/Timer-S1`` Hugging Face repository at commit
8911430cc7f32add5c8913afe12e3b05742f5bb2.

The generation script embedded below has a ``PRETRAINED`` switch, so anyone
with roughly 40 GB of RAM can regenerate references from the pretrained
checkpoint and add a corresponding test case.
"""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timer_s1 import TimerS1Forecaster
from sktime.tests.test_switch import run_test_for_class

# Reference forecasts were generated on CPU from the upstream Timer-S1 remote
# code in the bytedance-research/Timer-S1 Hugging Face repository at the
# immutable commit 8911430cc7f32add5c8913afe12e3b05742f5bb2, loaded through
# ``transformers.AutoModelForCausalLM`` with ``trust_remote_code=True`` and
# bypassing sktime entirely.
#
# The only pretrained Timer-S1 checkpoint has 8.3B parameters (16.6 GB in
# bfloat16), so the references use a small randomly initialized model instead.
# Weights are initialized from ``torch.manual_seed(_TIMER_S1_SEED)`` right
# before model construction, by both the reference script and the tests below.
# The vendored ``sktime.libs.timer_s1`` code registers parameters in the same
# order as upstream, so the seeded weights are identical.
#
# Environment: Python 3.12.14, torch 2.14.0, transformers 4.57.6, numpy 2.5.3.
#
# The forecast horizon of 48 exceeds the 32 values produced per generation
# step (input_token_len=16 times 1 + num_mtp_tokens), so two autoregressive
# generation steps are exercised, including the KV-cache update path.
#
# The reference values were produced by the following standalone script:
#
#     import numpy as np
#     import torch
#     from transformers import AutoConfig, AutoModelForCausalLM
#
#     # sktime is only used as a data source here (airline passengers).
#     from sktime.datasets import load_airline
#
#     REPO = "bytedance-research/Timer-S1"
#     REVISION = "8911430cc7f32add5c8913afe12e3b05742f5bb2"
#     SEED = 0
#     CONTEXT_LENGTH = 132  # airline minus last 12 months
#     HORIZON = 48  # > 32 values per generation step -> two generation steps
#     ALPHA = [0.1, 0.5, 0.9]
#
#     # PRETRAINED=False: small model with seeded random weights (the references
#     # in this file). PRETRAINED=True: the 8.3B checkpoint at REVISION loaded in
#     # float32 on CPU (the torch default device), which needs roughly 40 GB of
#     # RAM. Its default config has num_mtp_tokens=16, so HORIZON=48 is then
#     # covered by a single generation step. The matching sktime call is
#     # TimerS1Forecaster(model_path=REPO, forward_kwargs=..., deterministic=True)
#     # without setting any seed.
#     PRETRAINED = False
#
#     BASE_CONFIG = {
#         "hidden_size": 32,
#         "intermediate_size": 32,
#         "num_attention_heads": 4,
#         "num_experts": 4,
#         "num_hidden_layers": 2,
#         "num_mtp_tokens": 1,
#         "dropout_rate": 0.0,
#     }
#
#     CASES = {
#         "default": {"config": {}, "generate_kwargs": {}},
#         "revin-false": {"config": {}, "generate_kwargs": {"revin": False}},
#         "no-kv-cache": {"config": {"use_cache": False}, "generate_kwargs": {}},
#     }
#
#
#     def build_model(config_overrides):
#         if PRETRAINED:
#             config = AutoConfig.from_pretrained(
#                 REPO, revision=REVISION, trust_remote_code=True, **config_overrides
#             )
#             model = AutoModelForCausalLM.from_pretrained(
#                 REPO,
#                 config=config,
#                 revision=REVISION,
#                 trust_remote_code=True,
#                 dtype=torch.float32,
#             )
#             return model.eval()
#         config = AutoConfig.from_pretrained(
#             REPO,
#             revision=REVISION,
#             trust_remote_code=True,
#             **BASE_CONFIG,
#             **config_overrides,
#         )
#         torch.manual_seed(SEED)
#         model = AutoModelForCausalLM.from_config(
#             config, trust_remote_code=True, dtype=torch.float32
#         )
#         return model.eval()
#
#
#     def point_forecast_weights(quantiles):
#         # weights of the sktime point forecast: each quantile level is
#         # weighted by the width of its Voronoi cell on [0, 1]
#         quantiles = np.asarray(quantiles)
#         mids = (quantiles[:-1] + quantiles[1:]) / 2
#         edges = np.concatenate([[0.0], mids, [1.0]])
#         return np.diff(edges)
#
#
#     def fmt(values):
#         return "[" + ", ".join(f"{float(v):.9g}" for v in values) + "]"
#
#
#     def main():
#         y = load_airline().to_numpy()[:CONTEXT_LENGTH]
#         past_values = torch.tensor(y[None], dtype=torch.float32)
#
#         for name, case in CASES.items():
#             model = build_model(case["config"])
#             with torch.no_grad():
#                 output = model.generate(
#                     past_values, max_new_tokens=HORIZON, **case["generate_kwargs"]
#                 )
#             # output: [batch=1, n_quantiles=9, HORIZON]
#             raw = output[0].float().numpy()
#             quantiles = [round(q, 3) for q in model.config.quantiles]
#             weights = point_forecast_weights(quantiles)
#             point = np.average(raw, weights=weights, axis=0)
#             print(f"== {name}")
#             print("  point head :", fmt(point[:3]))
#             print("  point tail :", fmt(point[-3:]))
#             for a in ALPHA:
#                 print(f"  q={a} head :", fmt(raw[quantiles.index(a), :3]))
#
#
#     if __name__ == "__main__":
#         main()

_TIMER_S1_SEED = 0
_TIMER_S1_HORIZON = 48

# a config distinct from ``get_test_params`` so that the multiton model cache
# used by ``TimerS1Forecaster`` cannot hand back a model that was randomly
# initialized elsewhere in the test session with a different RNG state
_TIMER_S1_BASE_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 32,
    "num_attention_heads": 4,
    "num_experts": 4,
    "num_hidden_layers": 2,
    "num_mtp_tokens": 1,
    # randomly initialized models are left in training mode, so dropout
    # would otherwise be active in predict and depend on the RNG state
    "dropout_rate": 0.0,
}

# (config overrides, forward_kwargs, expected point head, expected point tail)
_TIMER_S1_UPSTREAM_REFERENCE_CASES = [
    pytest.param(
        {},
        None,
        [259.501067, 265.958752, 264.10755],
        [263.565356, 258.812035, 261.972707],
        id="default",
    ),
    pytest.param(
        {},
        {"revin": False},
        [-0.0773399752, 0.0199278594, 0.000543895178],
        [0.0119642884, -0.0347806784, -0.0110267563],
        id="revin-false",
    ),
    pytest.param(
        {"use_cache": False},
        None,
        [259.501067, 265.958752, 264.10755],
        [263.565353, 258.812035, 261.972708],
        id="no-kv-cache",
    ),
]

# native quantile forecasts of the "default" case, first three steps
_TIMER_S1_UPSTREAM_QUANTILE_HEAD = {
    0.1: [270.20993, 268.108398, 263.512726],
    0.5: [274.391785, 281.542847, 273.148865],
    0.9: [260.220581, 272.270081, 264.433838],
}


pytestmark = pytest.mark.skipif(
    not run_test_for_class(TimerS1Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _fit_seeded_forecaster(config_overrides, forward_kwargs):
    """Fit a TimerS1Forecaster with seeded random weights on airline data."""
    import torch

    y = load_airline()
    y_train = y.iloc[:-12]
    forecaster = TimerS1Forecaster(
        model_path=None,
        config={**_TIMER_S1_BASE_CONFIG, **config_overrides},
        forward_kwargs=forward_kwargs,
        deterministic=True,
    )
    # weights are drawn during fit, so the seed must be set right before it
    torch.manual_seed(_TIMER_S1_SEED)
    return forecaster.fit(y_train)


@pytest.mark.parametrize(
    "config_overrides,forward_kwargs,expected_head,expected_tail",
    _TIMER_S1_UPSTREAM_REFERENCE_CASES,
)
def test_timer_s1_airline_predictions_match_upstream_reference(
    config_overrides, forward_kwargs, expected_head, expected_tail
):
    """TimerS1 point predictions match the upstream Timer-S1 model outputs."""
    fh = np.arange(1, _TIMER_S1_HORIZON + 1)
    forecaster = _fit_seeded_forecaster(config_overrides, forward_kwargs)

    y_pred = forecaster.predict(fh=fh)

    assert len(y_pred) == _TIMER_S1_HORIZON
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


def test_timer_s1_airline_quantiles_match_upstream_reference():
    """TimerS1 quantile predictions match the upstream Timer-S1 model outputs."""
    fh = np.arange(1, _TIMER_S1_HORIZON + 1)
    alpha = sorted(_TIMER_S1_UPSTREAM_QUANTILE_HEAD)
    forecaster = _fit_seeded_forecaster({}, None)

    pred_quantiles = forecaster.predict_quantiles(fh=fh, alpha=alpha)

    assert pred_quantiles.shape == (_TIMER_S1_HORIZON, len(alpha))
    for a in alpha:
        # columns are (variable name, alpha), select by alpha only
        pred_a = pred_quantiles.xs(a, axis=1, level=1).iloc[:3, 0]
        np.testing.assert_allclose(
            pred_a.to_numpy(),
            np.asarray(_TIMER_S1_UPSTREAM_QUANTILE_HEAD[a], dtype=np.float32),
            rtol=1e-5,
            atol=1e-4,
        )
