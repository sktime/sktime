"""Golden-output regression tests for TimeLLMForecaster."""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.time_llm import TimeLLMForecaster
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(TimeLLMForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)

# Reference values come from the original Time-LLM code, KimMeen/Time-LLM at
# commit b13e881f86cd0475ce1b72c17110430663334955 (models/TimeLLM.py), called
# directly without sktime. The generator script is in the description of
# sktime PR #11118.
#
# Time-LLM has no trained forecasting checkpoint. Only the LLM backbone is
# pretrained, the patch embedding, mapping, reprogramming and output layers are
# randomly initialised. Both sides therefore seed torch with 0 right before the
# model is built, then cast it to bfloat16, put it in eval mode on CPU, and feed
# the last seq_len training values of load_airline (12-step holdout) as a
# float32 tensor of shape (1, seq_len, 1).
#
# Backbones as resolved from main when generated, since the estimator has no
# revision argument: openai-community/gpt2 @ 607a30d783dfa663caf39e06633721c8d4
# cfcd7e, google-bert/bert-base-uncased @ 86b5e0934494bd15c9632b12f734a8a67f723594
#
# Generated with python 3.13, torch 2.12.1 and transformers 5.13.0. The seeded
# weights are the same for transformers 4.40.2 to 5.13.0. transformers 4.31.0,
# upstream's pin, draws random numbers differently while loading the backbone,
# so it gives other weights and cannot be compared.
_TIME_LLM_UPSTREAM_CASES = [
    pytest.param(
        # defaults: seq_len=96 cuts the 132 training values, sparse horizon
        {},
        [1, 2, 3, 6, 12],
        [452.363464355, 263.309082031, 245.413604736, 253.209655762, 337.194458008],
        id="gpt2-default-sparse-fh",
    ),
    pytest.param(
        # BERT adds a [PAD] token, full history as context, other patching
        {
            "llm_model": "BERT",
            "seq_len": 132,
            "patch_len": 12,
            "stride": 6,
            "llm_layers": 2,
        },
        list(range(1, 13)),
        [228.883392334, 235.937149048, 256.890930176, 214.153503418, 129.716003418],
        id="bert-seq132-p12-s6-l2",
    ),
]

# Upstream runs the model in bfloat16, which keeps about 3 significant digits.
# The same code with torch 2.2.2 and transformers 4.40.2 gives the same weights,
# but bfloat16 kernels round differently: over the 24 reference values the
# relative difference to torch 2.12.1 has a root mean square of 2.1e-3 and a
# maximum of 5.1e-3. rtol is about six times the root mean square. An adapter
# bug changes the forecasts entirely, by far more than this.
_RTOL = 1.3e-2


@pytest.mark.parametrize("params,fh,expected", _TIME_LLM_UPSTREAM_CASES)
def test_time_llm_airline_predictions_match_upstream(params, fh, expected):
    """TimeLLMForecaster predictions match the original Time-LLM code.

    Each case uses its own backbone, so parallel workers never download the same
    unpinned Hugging Face repo at once.
    """
    import torch

    y_train = load_airline().iloc[:-12]
    forecaster = TimeLLMForecaster(device="cpu", **params)
    # weights outside the backbone are randomly initialised when built in fit
    torch.manual_seed(0)
    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict()

    # only the first values are stored
    actual = y_pred.to_numpy().ravel()[: len(expected)]
    np.testing.assert_allclose(actual, expected, rtol=_RTOL)
