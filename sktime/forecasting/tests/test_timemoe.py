# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for the TimeMoE forecaster.

The reference outputs in this module are generated from the pretrained
``Maple728/TimeMoE-50M`` (113M parameters) and ``Maple728/TimeMoE-200M``
(453M parameters) checkpoints, which fit comfortably in memory (below 3 GB
for both in float32), so no seeded-small-model fallback is used.

The remote code of the checkpoints, and hence ``TimeMoEForecaster``, was
written against ``transformers`` 4.40.1, which is also the upper bound of the
estimator's dependency pin. The references were therefore generated, and the
tests below run, with transformers 4.40.1.
"""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timemoe import TimeMoEForecaster
from sktime.tests.test_switch import run_test_for_class

# Reference forecasts were generated on CPU from the upstream TimeMoE remote
# code in the Maple728/TimeMoE-50M and Maple728/TimeMoE-200M Hugging Face
# repositories at the immutable commits
# 446753ee48ff3726d0606a81d0092d54acee995e (50M) and
# 794591bfeb1225fdf742cec0f4c71f20c3f3b87e (200M), model code and weights,
# loaded through ``transformers.AutoModelForCausalLM`` with
# ``trust_remote_code=True`` and bypassing sktime entirely.
#
# ``TimeMoEForecaster`` has no ``revision`` parameter, so the tests below
# download the checkpoints at the commits above with ``snapshot_download`` and
# pass the local snapshot directory as ``model_path``. The model code itself
# comes from the vendored copy in ``sktime.libs.timemoe``. The local path also
# keeps the cached model separate from the one ``get_test_params`` loads,
# which ``pretrain`` in the estimator suite fine-tunes in place.
#
# Environment: the horizon cases were generated on arm64 (macOS) with
# Python 3.12.14, torch 2.14.0, transformers 4.40.1, accelerate 0.28.0 and
# numpy 1.26.4. The sparse ``fh`` and short context cases were generated on
# x86-64 (linux) with Python 3.12.13, torch 2.9.1+cpu, transformers 4.40.1,
# accelerate 0.28.0 and numpy 1.26.4, where the script also reproduces the
# horizon cases within the tolerances below.
#
# The reference values were produced by the following standalone script:
#
#     """Generate golden reference outputs for TimeMoEForecaster from Time-MoE.
#
#     Calls the upstream remote code of the Maple728/TimeMoE checkpoints directly
#     through transformers, as in the README of github.com/Time-MoE/Time-MoE,
#     and never imports sktime.
#     """
#
#     import platform
#     from importlib.metadata import version
#
#     import numpy as np
#     import torch
#     from transformers import AutoModelForCausalLM
#
#     # Hugging Face repository -> immutable commit (model code and weights)
#     REVISIONS = {
#         "Maple728/TimeMoE-50M": "446753ee48ff3726d0606a81d0092d54acee995e",
#         "Maple728/TimeMoE-200M": "794591bfeb1225fdf742cec0f4c71f20c3f3b87e",
#     }
#     SEED = 0
#
#     # sktime.datasets.load_airline, monthly from 1949-01
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
#     ]  # fmt: skip
#
#     # (repo, fh, context length). The checkpoints have output heads for 1, 8, 32
#     # and 64 steps; each generation step uses the largest head that does not
#     # exceed the number of values still to be generated.
#     # * fh 1..1: one step of the 1-step head
#     # * fh 1..3: three autoregressive steps of the 1-step head
#     # * fh 1..12: the 8-step head, then four steps of the 1-step head
#     # * fh 1..100: the 64-step and 32-step heads, then four 1-step steps
#     # * fh [2, 5, 12]: sparse, generates 12 steps and selects three
#     # * context 36: a shorter context than the full training series
#     CASES = [
#         (repo, list(range(1, horizon + 1)), None)
#         for repo in REVISIONS
#         for horizon in (1, 3, 12, 100)
#     ] + [
#         ("Maple728/TimeMoE-50M", [2, 5, 12], None),
#         ("Maple728/TimeMoE-50M", list(range(1, 13)), 36),
#     ]
#
#
#     def fmt(values):
#         return "[" + ", ".join(f"{float(v):.9g}" for v in values) + "]"
#
#
#     def forecast(model, fh, context_length):
#         # the first 132 values, i.e., without the last 12 months
#         y = np.asarray(AIRLINE_VALUES, dtype=np.float32)[:-12]
#         if context_length is not None:
#             y = y[-context_length:]
#         horizon = max(fh)
#         # z-score the context, generate, invert the z-score;
#         # a new tensor per call, since generate modifies its input
#         seqs = torch.tensor(y, dtype=torch.float32)[None]
#         mean = seqs.mean(dim=-1, keepdim=True)
#         std = seqs.std(dim=-1, keepdim=True)
#         normed_seqs = (seqs - mean) / std
#         torch.manual_seed(SEED)
#         with torch.no_grad():
#             output = model.generate(normed_seqs, max_new_tokens=horizon)
#         # output: [batch=1, context length + horizon]
#         values = (output[:, -horizon:] * std + mean)[0].numpy()
#         return values[[h - 1 for h in fh]]
#
#
#     def main():
#         print("python", platform.python_version())
#         for package in ["numpy", "torch", "transformers", "accelerate"]:
#             print(package, version(package))
#
#         models = {}
#         for repo, revision in REVISIONS.items():
#             # CPU, float32; from_pretrained puts the model in eval mode
#             models[repo] = AutoModelForCausalLM.from_pretrained(
#                 repo,
#                 revision=revision,
#                 trust_remote_code=True,
#                 device_map="cpu",
#                 torch_dtype=torch.float32,
#             )
#             models[repo].eval()
#
#         for repo, fh, context_length in CASES:
#             first = forecast(models[repo], fh, context_length)
#             second = forecast(models[repo], fh, context_length)
#             assert np.array_equal(first, second), "not reproducible"
#             print(f"== {repo} fh {fh[0]}..{fh[-1]} ({len(fh)}) ctx {context_length}")
#             print("  head :", fmt(first[:3]))
#             print("  tail :", fmt(first[-3:]))
#
#
#     if __name__ == "__main__":
#         main()


# (model_path, horizon, expected head, expected tail)
# head and tail are the first and last three values of the forecast,
# or the entire forecast if the horizon is shorter than three
_TIMEMOE_UPSTREAM_REFERENCE_CASES = [
    pytest.param(
        "Maple728/TimeMoE-50M",
        1,
        [409.940826],
        [409.940826],
        id="50M-horizon-1",
    ),
    pytest.param(
        "Maple728/TimeMoE-50M",
        3,
        [409.940826, 384.68634, 434.162842],
        [409.940826, 384.68634, 434.162842],
        id="50M-horizon-3",
    ),
    pytest.param(
        "Maple728/TimeMoE-50M",
        12,
        [409.770081, 388.648621, 429.434265],
        [416.300995, 369.300354, 397.19635],
        id="50M-horizon-12",
    ),
    pytest.param(
        "Maple728/TimeMoE-50M",
        100,
        [410.360504, 388.51532, 429.737488],
        [243.106781, 259.304993, 256.562927],
        id="50M-horizon-100",
    ),
    pytest.param(
        "Maple728/TimeMoE-200M",
        1,
        [402.947571],
        [402.947571],
        id="200M-horizon-1",
    ),
    pytest.param(
        "Maple728/TimeMoE-200M",
        3,
        [402.947571, 387.854919, 428.837341],
        [402.947571, 387.854919, 428.837341],
        id="200M-horizon-3",
    ),
    pytest.param(
        "Maple728/TimeMoE-200M",
        12,
        [402.694092, 389.358459, 418.003601],
        [366.982117, 325.898743, 357.255096],
        id="200M-horizon-12",
    ),
    pytest.param(
        "Maple728/TimeMoE-200M",
        100,
        [402.583557, 388.632568, 419.006714],
        [203.24472, 222.443176, 220.659164],
        id="200M-horizon-100",
    ),
]


# Tolerances. The references were generated on arm64 (macOS). TimeMoE runs in
# float32 and feeds its own output back in as context, so differences between
# the BLAS kernels of different CPU architectures accumulate along the
# forecast: on x86-64 (linux and windows), the values around step 100 of the
# 200M model's forecast deviate from the arm64 references by a relative 6.4e-5
# at most, and by slightly different amounts on different x86-64 machines.
# The looser tolerance therefore applies only to the tail of long horizons,
# and is about 8 times the largest observed drift; heads, and the tails of
# short horizons, are compared at the default tolerance.
#
# The drift is not stochasticity. Prediction is deterministic: the checkpoints
# have no dropout (``attention_dropout=0.0``, no ``nn.Dropout`` modules),
# ``from_pretrained`` puts the model in eval mode, the checkpoints' generation
# config has ``do_sample=False``, and the vendored ``TSGenerationMixin`` only
# overrides ``_greedy_search``, which uses the regression output directly
# rather than sampling from it. On one machine, repeated runs of each case
# below, with ``seed=0``, with ``seed=None``, and calling ``generate``
# directly without touching the RNG, were bit-identical. Fixing a seed
# therefore cannot remove the mismatch; only the per-architecture tolerance
# can. CI showed the same split: before the tolerance was added, all arm64
# (macOS) jobs passed and the x86-64 (ubuntu, windows) jobs failed, by the
# relative 6.4e-5 quoted above at most.
_RTOL = 1e-5
_RTOL_LONG_HORIZON_TAIL = 5e-4
_LONG_HORIZON = 100


# Hugging Face repository -> immutable commit (model code and weights)
_REVISIONS = {
    "Maple728/TimeMoE-50M": "446753ee48ff3726d0606a81d0092d54acee995e",
    "Maple728/TimeMoE-200M": "794591bfeb1225fdf742cec0f4c71f20c3f3b87e",
}


def _pinned_snapshot(repo_id):
    """Return the local directory of ``repo_id`` at its pinned commit."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=_REVISIONS[repo_id])


pytestmark = pytest.mark.skipif(
    not run_test_for_class(TimeMoEForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


@pytest.mark.parametrize(
    "model_path,horizon,expected_head,expected_tail",
    _TIMEMOE_UPSTREAM_REFERENCE_CASES,
)
def test_timemoe_predictions_match_upstream_reference(
    model_path, horizon, expected_head, expected_tail
):
    """TimeMoE predictions match the upstream Maple728/TimeMoE model outputs."""
    # airline passengers without the last 12 months
    y_train = load_airline().iloc[:-12]
    fh = np.arange(1, horizon + 1)
    forecaster = TimeMoEForecaster(
        model_path=_pinned_snapshot(model_path), device="cpu", seed=0
    )

    y_pred = forecaster.fit(y_train, fh=fh).predict(fh=fh)

    assert len(y_pred) == horizon
    np.testing.assert_allclose(
        y_pred.iloc[:3].to_numpy().ravel(),
        np.asarray(expected_head, dtype=np.float32),
        rtol=_RTOL,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        y_pred.iloc[-3:].to_numpy().ravel(),
        np.asarray(expected_tail, dtype=np.float32),
        rtol=_RTOL_LONG_HORIZON_TAIL if horizon >= _LONG_HORIZON else _RTOL,
        atol=1e-4,
    )


# (model_path, fh, context length, expected forecast at fh)
_TIMEMOE_UPSTREAM_FH_CONTEXT_CASES = [
    # sparse fh: generates 12 steps and returns only the requested ones
    pytest.param(
        "Maple728/TimeMoE-50M",
        [2, 5, 12],
        None,
        [388.648346, 433.013794, 397.196289],
        id="50M-sparse-fh",
    ),
    # the last 36 values of the training series as context;
    # first and last three values of the 12-step forecast
    pytest.param(
        "Maple728/TimeMoE-50M",
        [1, 2, 3, 10, 11, 12],
        36,
        [414.354523, 410.816223, 449.615448, 423.824585, 382.559784, 416.8862],
        id="50M-context-36",
    ),
]


@pytest.mark.parametrize(
    "model_path,fh,context_length,expected",
    _TIMEMOE_UPSTREAM_FH_CONTEXT_CASES,
)
def test_timemoe_fh_and_context_match_upstream_reference(
    model_path, fh, context_length, expected
):
    """TimeMoE matches upstream for sparse fh and a shorter context."""
    # airline passengers without the last 12 months
    y_train = load_airline().iloc[:-12]
    if context_length is not None:
        y_train = y_train.iloc[-context_length:]
    forecaster = TimeMoEForecaster(
        model_path=_pinned_snapshot(model_path), device="cpu", seed=0
    )

    y_pred = forecaster.fit(y_train).predict(fh=fh)

    expected_index = y_train.index[-1] + np.asarray(fh)
    assert list(y_pred.index) == list(expected_index)
    np.testing.assert_allclose(
        y_pred.to_numpy().ravel(),
        np.asarray(expected, dtype=np.float32),
        rtol=_RTOL,
        atol=1e-4,
    )
