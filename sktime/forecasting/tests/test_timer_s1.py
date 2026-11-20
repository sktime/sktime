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
"""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timer_s1 import TimerS1Forecaster
from sktime.tests.test_switch import run_test_for_class

# Reference forecasts were generated on CPU from the upstream Timer-S1 remote
# code in the bytedance-research/Timer-S1 Hugging Face repository at the
# immutable commit 8911430cc7f32add5c8913afe12e3b05742f5bb2, loaded through
# ``transformers.AutoModelForCausalLM`` with ``trust_remote_code=True``, in a
# venv without sktime. The airline values are embedded in the script.
#
# The only pretrained Timer-S1 checkpoint has 8.3B parameters (16.6 GB in
# bfloat16), so the references use a small randomly initialized model instead.
# Weights are initialized from ``torch.manual_seed(_TIMER_S1_SEED)`` right
# before model construction, by both the reference script and the tests below.
# The vendored ``sktime.libs.timer_s1`` code registers parameters in the same
# order as upstream, so the seeded weights are identical. No checkpoint is
# downloaded by the tests.
#
# Environment: Python 3.12.13, torch 2.15.0 (CPU), transformers 4.57.6,
# numpy 2.5.5. Two runs gave identical values. The tests also pass with
# transformers 5.
#
# A forecast horizon of 48 exceeds the 32 values produced per generation step
# (input_token_len=16 times 1 + num_mtp_tokens), so two autoregressive
# generation steps are exercised, including the KV-cache update path.
#
# The reference values were produced by the following standalone script:
#
#     import numpy as np
#     import torch
#     from huggingface_hub import snapshot_download
#     from transformers import AutoConfig, AutoModelForCausalLM
#
#     REPO = "bytedance-research/Timer-S1"
#     REVISION = "8911430cc7f32add5c8913afe12e3b05742f5bb2"
#     SEED = 0
#
#     # values of sktime.datasets.load_airline, monthly from 1949-01
#     AIRLINE = [
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
#     # small config, weights drawn from torch.manual_seed(SEED); every other
#     # setting comes from the checkpoint's config.json at REVISION
#     BASE_CONFIG = {
#         "hidden_size": 32,
#         "intermediate_size": 32,
#         "num_attention_heads": 4,
#         "num_experts": 4,
#         "num_hidden_layers": 2,
#         "num_mtp_tokens": 1,  # 16 + 16 = 32 values per generation step
#     }
#     FH_48 = list(range(1, 49))  # 48 > 32: two generation steps
#
#     # name: (n training values, config overrides, generate kwargs, fh)
#     CASES = {
#         # 132 values, left padded by the model to 9 patches of 16
#         "default": (132, {}, {}, FH_48),
#         "revin-false": (132, {}, {"revin": False}, FH_48),
#         "no-kv-cache": (132, {"use_cache": False}, {}, FH_48),
#         # 128 values, a whole number of patches, no padding
#         "unpadded": (128, {}, {}, FH_48),
#         # max(fh) = 40: second step returns only 8 of 16 values
#         "sparse-fh": (132, {}, {}, [2, 17, 40]),
#     }
#     ALPHA = [0.1, 0.5, 0.9]
#
#
#     def build_model(config_overrides):
#         # config.json and the remote model code only, no weights
#         path = snapshot_download(
#             REPO, revision=REVISION, allow_patterns=["*.py", "config.json"]
#         )
#         config = AutoConfig.from_pretrained(
#             path, trust_remote_code=True, **BASE_CONFIG, **config_overrides
#         )
#         torch.manual_seed(SEED)
#         model = AutoModelForCausalLM.from_config(
#             config, trust_remote_code=True, dtype=torch.float32
#         )
#         return model.eval()
#
#
#     def point_weights(quantiles):
#         # sktime point forecast: quantiles weighted by their Voronoi cell
#         q = np.asarray(quantiles)
#         edges = np.concatenate([[0.0], (q[:-1] + q[1:]) / 2, [1.0]])
#         return np.diff(edges)
#
#
#     def fmt(values):
#         return "[" + ", ".join(f"{float(v):.9g}" for v in values) + "]"
#
#
#     for name, (n, config_overrides, gen_kwargs, fh) in CASES.items():
#         model = build_model(config_overrides)
#         y = np.asarray(AIRLINE[:n], dtype=np.float64)
#         past = torch.tensor(y[None], dtype=torch.float32)
#         idx = np.asarray(fh) - 1
#         with torch.no_grad():
#             out = model.generate(past, max_new_tokens=max(fh), **gen_kwargs)
#         raw = out[0].float().numpy()  # [n_quantiles, max(fh)]
#         quantiles = [round(q, 3) for q in model.config.quantiles]
#         point = np.average(raw, weights=point_weights(quantiles), axis=0)[idx]
#         print(name, fmt(point[:3]), fmt(point[-3:]))
#         if name == "default":
#             for a in ALPHA:
#                 q_a = raw[quantiles.index(a)]
#                 print(f"  q={a}", fmt(q_a[:3]), fmt(q_a[-3:]))

_TIMER_S1_SEED = 0

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
}

_FH_48 = list(range(1, 49))

# (n training values, config overrides, forward_kwargs, fh,
#  expected point forecasts at selected fh steps)
_TIMER_S1_UPSTREAM_REFERENCE_CASES = [
    pytest.param(
        132,
        {},
        None,
        _FH_48,
        {
            1: 259.501064,
            2: 265.958755,
            3: 264.10755,
            46: 263.565353,
            47: 258.812035,
            48: 261.972707,
        },
        id="default",
    ),
    pytest.param(
        132,
        {},
        {"revin": False},
        _FH_48,
        {
            1: -0.0773400117,
            2: 0.0199278687,
            3: 0.000543892756,
            46: 0.011964298,
            47: -0.0347806731,
            48: -0.0110267494,
        },
        id="revin-false",
    ),
    pytest.param(
        132,
        {"use_cache": False},
        None,
        _FH_48,
        {
            1: 259.501064,
            2: 265.958755,
            3: 264.10755,
            46: 263.565353,
            47: 258.812035,
            48: 261.972707,
        },
        id="no-kv-cache",
    ),
]

# native quantile forecasts of the "default" case at selected fh steps
_TIMER_S1_UPSTREAM_QUANTILES = {
    0.1: {
        1: 270.2099,
        2: 268.108398,
        3: 263.512726,
    },
    0.5: {
        1: 274.391785,
        2: 281.542847,
        3: 273.148865,
    },
    0.9: {
        1: 260.220581,
        2: 272.270081,
        3: 264.433807,
    },
}


pytestmark = pytest.mark.skipif(
    not run_test_for_class(TimerS1Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _fit_seeded_forecaster(n_train, config_overrides, forward_kwargs):
    """Fit a TimerS1Forecaster with seeded random weights on airline data."""
    import torch

    y_train = load_airline().iloc[:n_train]
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
    "n_train,config_overrides,forward_kwargs,fh,expected",
    _TIMER_S1_UPSTREAM_REFERENCE_CASES,
)
def test_timer_s1_airline_predictions_match_upstream_reference(
    n_train, config_overrides, forward_kwargs, fh, expected
):
    """TimerS1 point predictions match the upstream Timer-S1 model outputs."""
    forecaster = _fit_seeded_forecaster(n_train, config_overrides, forward_kwargs)

    y_pred = forecaster.predict(fh=fh)

    assert len(y_pred) == len(fh)
    positions = [fh.index(step) for step in expected]
    np.testing.assert_allclose(
        y_pred.iloc[positions].to_numpy(),
        np.asarray(list(expected.values()), dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )


def test_timer_s1_airline_quantiles_match_upstream_reference():
    """TimerS1 quantile predictions match the upstream Timer-S1 model outputs."""
    alpha = sorted(_TIMER_S1_UPSTREAM_QUANTILES)
    forecaster = _fit_seeded_forecaster(132, {}, None)

    pred_quantiles = forecaster.predict_quantiles(fh=_FH_48, alpha=alpha)

    assert pred_quantiles.shape == (len(_FH_48), len(alpha))
    for a in alpha:
        expected = _TIMER_S1_UPSTREAM_QUANTILES[a]
        positions = [_FH_48.index(step) for step in expected]
        # columns are (variable name, alpha), select by alpha only
        pred_a = pred_quantiles.xs(a, axis=1, level=1).iloc[positions, 0]
        np.testing.assert_allclose(
            pred_a.to_numpy(),
            np.asarray(list(expected.values()), dtype=np.float32),
            rtol=1e-5,
            atol=1e-4,
            err_msg=f"alpha={a}",
        )
