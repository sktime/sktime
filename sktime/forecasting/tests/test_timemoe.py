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
# ``TimeMoEForecaster`` has no ``revision`` parameter, so the tests below load
# the ``main`` revision of the checkpoints, which at the time of writing are
# the commits above; the model code itself comes from the vendored copy in
# ``sktime.libs.timemoe``, which is functionally identical to those commits.
#
# Environment: Python 3.12.14, torch 2.14.0, transformers 4.40.1,
# accelerate 0.28.0, numpy 1.26.4.
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
#     # Hugging Face repository -> immutable commit (model code and weights)
#     REVISIONS = {
#         "Maple728/TimeMoE-50M": "446753ee48ff3726d0606a81d0092d54acee995e",
#         "Maple728/TimeMoE-200M": "794591bfeb1225fdf742cec0f4c71f20c3f3b87e",
#     }
#     SEED = 0
#
#     # The released checkpoints have output heads for 1, 8, 32 and 64 steps;
#     # each generation step uses the largest head that does not exceed the
#     # number of values still to be generated (the 1-step head is cut to size).
#     #
#     # * 1: a single step of the 1-step head.
#     # * 3: three autoregressive steps of the 1-step head.
#     # * 12: one step of the 8-step head, then four steps of the 1-step head.
#     # * 100: the 64-step and 32-step heads, then four steps of the 1-step head.
#     HORIZONS = [1, 3, 12, 100]
#
#
#     def fmt(values):
#         return "[" + ", ".join(f"{float(v):.9g}" for v in values) + "]"
#
#
#     def main():
#         # first 132 airline values, i.e., without the last 12 months
#         y = load_airline().to_numpy()[:-12].astype(np.float32)
#
#         for repo, revision in REVISIONS.items():
#             # loaded on CPU in float32. The model is deterministic: the
#             # checkpoints have no dropout and generation is greedy, so SEED
#             # only pins the RNG state for good measure.
#             model = AutoModelForCausalLM.from_pretrained(
#                 repo, revision=revision, trust_remote_code=True, device_map="cpu"
#             )
#             model.eval()
#
#             for horizon in HORIZONS:
#                 # usage as in the README of github.com/Time-MoE/Time-MoE:
#                 # z-score the context, generate, invert the z-score.
#                 # A new tensor per call, since generate modifies its input.
#                 seqs = torch.tensor(y, dtype=torch.float32)[None]
#                 mean = seqs.mean(dim=-1, keepdim=True)
#                 std = seqs.std(dim=-1, keepdim=True)
#                 normed_seqs = (seqs - mean) / std
#                 torch.manual_seed(SEED)
#                 with torch.no_grad():
#                     output = model.generate(normed_seqs, max_new_tokens=horizon)
#                 # output: [batch=1, context length + horizon]
#                 forecast = (output[:, -horizon:] * std + mean)[0].numpy()
#                 assert forecast.shape == (horizon,), forecast.shape
#                 print(f"== {repo} horizon {horizon}")
#                 print("  head :", fmt(forecast[:3]))
#                 print("  tail :", fmt(forecast[-3:]))
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


# The references were generated on arm64 (macOS). TimeMoE runs in float32 and
# feeds its own output back in as context, so differences between the BLAS
# kernels of different CPU architectures accumulate over long horizons: on
# x86-64 (linux and windows) the tail of the 200M model's 100 step forecast
# deviates from the references by a relative 6.4e-5, and by slightly different
# amounts on different x86-64 machines. The tolerance leaves headroom for
# that, and is still orders of magnitude below the deviation caused by an
# actual change in the forecasting logic.
_RTOL = 1e-3


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
    forecaster = TimeMoEForecaster(model_path=model_path, device="cpu", seed=0)

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
        rtol=_RTOL,
        atol=1e-4,
    )
