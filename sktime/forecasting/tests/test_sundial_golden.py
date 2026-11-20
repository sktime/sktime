# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Golden-output regression tests for SundialForecaster against upstream Sundial.

The checkpoint's remote code, and hence ``SundialForecaster``, only supports
``transformers`` 4.40.x: newer releases removed the cache and generation
internals it relies on.
"""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.sundial import SundialForecaster
from sktime.tests.test_switch import run_test_for_class

# Reference forecasts were generated on CPU in float32 from the upstream Sundial
# remote code in the thuml/sundial-base-128m Hugging Face repository at the
# immutable commit 3212e42564493f520593e5414af4367fc4b49226 (model code and
# weights), loaded through ``transformers.AutoModelForCausalLM`` with
# ``trust_remote_code=True``, bypassing sktime entirely.
#
# ``SundialForecaster`` has no ``revision`` parameter, so the tests download
# that commit with ``huggingface_hub.snapshot_download`` and pass the local
# snapshot as ``model_path``. Model code in the tests comes from the vendored
# ``sktime.libs.sundial``.
#
# Input: the first 132 airline values, the last 12 held out. The
# "multivariate" case adds a second column drawn with
# ``np.random.default_rng(42).normal(200, 20, 132)``. Sundial samples paths by
# flow matching from Gaussian noise, so the torch RNG is seeded before every
# ``generate`` call with the seed ``SundialForecaster`` derives from
# ``random_state=0``. The point forecast is the mean over sample paths.
#
# "revin-false" disables Sundial's instance normalisation, so the model sees
# raw passenger counts and forecasts far from the data. That is upstream
# behaviour, pinned here only to cover the code path.
#
# Environment: Python 3.12.13, torch 2.15.0, transformers 4.40.1, numpy 1.26.4.
#
# Generation script (outputs below), run without sktime installed:
#
#     import numpy as np
#     import torch
#     from transformers import AutoModelForCausalLM
#
#     REPO = "thuml/sundial-base-128m"
#     REVISION = "3212e42564493f520593e5414af4367fc4b49226"
#     RANDOM_STATE = 0
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
#     ]
#
#     CASES = {
#         "default": {"horizon": 12},
#         "num-samples-20": {"horizon": 12, "kwargs": {"num_samples": 20}},
#         "revin-false": {"horizon": 12, "kwargs": {"revin": False}},
#         # 128 is a multiple of input_token_len=16, so no left padding
#         "context-128": {"horizon": 12, "context": 128},
#         "multivariate": {"horizon": 12, "multivariate": True},
#         "full-step-720": {"horizon": 720},
#     }
#
#     def make_input(context=None, multivariate=False):
#         y = np.asarray(AIRLINE_VALUES[:-12], dtype=np.float64)
#         if context is not None:
#             y = y[-context:]
#         series = [y]
#         if multivariate:
#             rng = np.random.default_rng(42)
#             series.append(rng.normal(loc=200.0, scale=20.0, size=len(y)))
#         return np.stack(series)
#
#     def torch_seed(random_state):
#         # the torch seed SundialForecaster derives from random_state
#         return np.random.RandomState(random_state).randint(np.iinfo(np.int32).max)
#
#     def run_case(model, horizon, kwargs=None, context=None, multivariate=False):
#         kwargs = kwargs or {}
#         past_values = torch.from_numpy(make_input(context, multivariate))
#         past_values = past_values.to(torch.float32)
#         torch.manual_seed(torch_seed(RANDOM_STATE))
#         with torch.no_grad():
#             output = model.generate(past_values, max_new_tokens=horizon, **kwargs)
#         # output: [n_series, num_samples, horizon]
#         return output.detach().float().cpu().numpy().mean(axis=1)
#
#     model = AutoModelForCausalLM.from_pretrained(
#         REPO, revision=REVISION, trust_remote_code=True, torch_dtype=torch.float32
#     )
#     model.eval()
#     for name, case in CASES.items():
#         forecast = run_case(model, **case)
#         print(name, forecast[:, :3], forecast[:, -3:])

_SUNDIAL_REPO = "thuml/sundial-base-128m"
_SUNDIAL_REVISION = "3212e42564493f520593e5414af4367fc4b49226"
_SUNDIAL_RANDOM_STATE = 0

# (forward_kwargs, horizon, context, multivariate, expected heads, expected tails)
# with one head and one tail of three values per series
_SUNDIAL_UPSTREAM_REFERENCE_CASES = [
    pytest.param(
        None,
        12,
        None,
        False,
        [[401.111633, 384.245575, 432.857788]],
        [[404.446808, 349.093719, 382.332642]],
        id="default",
    ),
    pytest.param(
        {"num_samples": 20},
        12,
        None,
        False,
        [[401.604553, 386.582733, 428.570251]],
        [[409.647369, 360.100647, 389.704773]],
        id="num-samples-20",
    ),
    pytest.param(
        {"revin": False},
        12,
        None,
        False,
        [[5.97387695, 5.56297064, 5.22011328]],
        [[3.93165517, 4.39560938, 3.50664186]],
        id="revin-false",
    ),
    pytest.param(
        None,
        12,
        128,
        False,
        [[399.372284, 385.054779, 436.487732]],
        [[407.936584, 352.984497, 386.854492]],
        id="context-128",
    ),
    pytest.param(
        None,
        12,
        None,
        True,
        [
            [401.111664, 384.245605, 432.857849],
            [198.157944, 203.88208, 206.268616],
        ],
        [
            [404.446777, 349.093689, 382.332581],
            [188.356201, 200.829117, 185.469543],
        ],
        id="multivariate",
    ),
    pytest.param(
        None,
        720,
        None,
        False,
        [[401.111633, 384.245575, 432.857788]],
        [[326.205933, 271.290344, 297.121765]],
        id="full-step-720",
    ),
]


pytestmark = pytest.mark.skipif(
    not run_test_for_class(SundialForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


@pytest.fixture(scope="module")
def sundial_checkpoint_path():
    """Local snapshot of the checkpoint at the pinned revision."""
    from huggingface_hub import snapshot_download

    return snapshot_download(_SUNDIAL_REPO, revision=_SUNDIAL_REVISION)


def _load_y_train(context, multivariate):
    """Airline passengers without the last 12 months, as the generator uses."""
    y = load_airline().iloc[:-12]
    if context is not None:
        y = y.iloc[-context:]
    if not multivariate:
        return y
    rng = np.random.default_rng(42)
    extra = rng.normal(loc=200.0, scale=20.0, size=len(y))
    return pd.DataFrame({"airline": y.to_numpy(), "extra": extra}, index=y.index)


@pytest.mark.parametrize(
    "forward_kwargs,horizon,context,multivariate,expected_heads,expected_tails",
    _SUNDIAL_UPSTREAM_REFERENCE_CASES,
)
def test_sundial_predictions_match_upstream_reference(
    sundial_checkpoint_path,
    forward_kwargs,
    horizon,
    context,
    multivariate,
    expected_heads,
    expected_tails,
):
    """Sundial predictions match the upstream thuml/sundial-base-128m outputs."""
    y_train = _load_y_train(context, multivariate)
    fh = np.arange(1, horizon + 1)
    forecaster = SundialForecaster(
        model_path=sundial_checkpoint_path,
        device="cpu",
        forward_kwargs=forward_kwargs,
        random_state=_SUNDIAL_RANDOM_STATE,
    )

    y_pred = forecaster.fit(y_train, fh=fh).predict(fh=fh)
    y_pred = pd.DataFrame(y_pred)

    assert y_pred.shape == (horizon, len(expected_heads))
    np.testing.assert_allclose(
        y_pred.iloc[:3].to_numpy().T,
        np.asarray(expected_heads, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        y_pred.iloc[-3:].to_numpy().T,
        np.asarray(expected_tails, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )
