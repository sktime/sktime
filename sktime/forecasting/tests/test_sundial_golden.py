# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for the Sundial forecaster against upstream outputs.

The reference outputs in this module are generated from the pretrained
``thuml/sundial-base-128m`` checkpoint (128M parameters), which fits
comfortably in memory, so no seeded-small-model fallback is used.

The checkpoint's remote code, and hence ``SundialForecaster``, only supports
``transformers`` 4.40.x: newer releases removed the cache and generation
internals it relies on. The references were therefore generated, and the
tests below run, with transformers 4.40.1, the version recorded in the
upstream checkpoint config.
"""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.sundial import SundialForecaster
from sktime.tests.test_switch import run_test_for_class

# Reference forecasts were generated on CPU from the upstream Sundial remote
# code in the thuml/sundial-base-128m Hugging Face repository at the immutable
# commit 3212e42564493f520593e5414af4367fc4b49226 (model code and weights),
# loaded through ``transformers.AutoModelForCausalLM`` with
# ``trust_remote_code=True`` and bypassing sktime entirely.
#
# ``SundialForecaster`` has no ``revision`` parameter, so the tests below load
# the ``main`` revision of the checkpoint, which at the time of writing is the
# commit above; the model code itself comes from the vendored copy in
# ``sktime.libs.sundial``, which is functionally identical to that commit for
# float32 on CPU.
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
#     REPO = "thuml/sundial-base-128m"
#     REVISION = "3212e42564493f520593e5414af4367fc4b49226"
#     RANDOM_STATE = 0
#
#     # The pretrained checkpoint at REVISION is loaded on CPU, the torch
#     # default device. Sundial is generative: each forecast is a sample path
#     # obtained by flow matching from Gaussian noise, so the torch RNG is
#     # seeded before every ``generate`` call.
#     #
#     # Each case mirrors one SundialForecaster configuration with
#     # random_state=0 on the first 132 airline values:
#     #
#     # * "default": SundialForecaster(random_state=0), fh=1..12. A single
#     #   sample path (num_samples=1) with instance normalisation (revin=True),
#     #   one generation step truncated to 12 values.
#     # * "num-samples-20": as "default" with
#     #   forward_kwargs={"num_samples": 20}; the point forecast is the mean
#     #   over the 20 sample paths.
#     # * "full-step-720": as "default" with fh=1..720, the full output of one
#     #   generation step (config.output_token_lens[-1] values) and the longest
#     #   horizon SundialForecaster accepts.
#     CASES = {
#         "default": {"num_samples": 1, "horizon": 12},
#         "num-samples-20": {"num_samples": 20, "horizon": 12},
#         "full-step-720": {"num_samples": 1, "horizon": 720},
#     }
#
#
#     def torch_seed(random_state):
#         # the torch seed that SundialForecaster derives from random_state
#         rng = np.random.RandomState(random_state)
#         return rng.randint(np.iinfo(np.int32).max)
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
#         y = load_airline().to_numpy()[:132].astype(np.float32)
#         past_values = torch.tensor(y, dtype=torch.float32)[None]
#
#         for name, case in CASES.items():
#             torch.manual_seed(torch_seed(RANDOM_STATE))
#             output = model.generate(
#                 past_values,
#                 max_new_tokens=case["horizon"],
#                 num_samples=case["num_samples"],
#             )
#             # output: [batch=1, num_samples, horizon]
#             assert output.shape == (1, case["num_samples"], case["horizon"])
#             forecast = output[0].mean(dim=0).numpy()
#             print(f"== {name}")
#             print("  head :", fmt(forecast[:3]))
#             print("  tail :", fmt(forecast[-3:]))
#
#
#     if __name__ == "__main__":
#         main()

_SUNDIAL_MODEL_PATH = "thuml/sundial-base-128m"
_SUNDIAL_RANDOM_STATE = 0


def _load_airline_train():
    """Airline passengers without the last 12 months."""
    return load_airline().iloc[:-12]


# (forward_kwargs, horizon, expected head, expected tail)
_SUNDIAL_UPSTREAM_REFERENCE_CASES = [
    pytest.param(
        None,
        12,
        [401.111694, 384.245636, 432.857819],
        [404.446838, 349.093811, 382.332764],
        id="default",
    ),
    pytest.param(
        {"num_samples": 20},
        12,
        [401.604523, 386.582764, 428.570251],
        [409.647522, 360.100769, 389.704956],
        id="num-samples-20",
    ),
    pytest.param(
        None,
        720,
        [401.111694, 384.245636, 432.857819],
        [326.205902, 271.290314, 297.121765],
        id="full-step-720",
    ),
]


pytestmark = pytest.mark.skipif(
    not run_test_for_class(SundialForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


@pytest.mark.parametrize(
    "forward_kwargs,horizon,expected_head,expected_tail",
    _SUNDIAL_UPSTREAM_REFERENCE_CASES,
)
def test_sundial_predictions_match_upstream_reference(
    forward_kwargs, horizon, expected_head, expected_tail
):
    """Sundial predictions match the upstream thuml/sundial-base-128m outputs."""
    y_train = _load_airline_train()
    fh = np.arange(1, horizon + 1)
    forecaster = SundialForecaster(
        model_path=_SUNDIAL_MODEL_PATH,
        device="cpu",
        forward_kwargs=forward_kwargs,
        random_state=_SUNDIAL_RANDOM_STATE,
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
