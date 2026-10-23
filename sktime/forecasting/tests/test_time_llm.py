"""Call-path parity tests for TimeLLMForecaster.

These tests compare ``TimeLLMForecaster.predict`` to a direct
``Model.forward`` on the vendored KimMeen/Time-LLM fork
(``sktime.libs.time_llm.TimeLLM.Model``) in the same process.

They are not a golden lock against a pinned KimMeen/Time-LLM checkout
or a public Time-LLM forecasting checkpoint. There is no installable
upstream package. ``TINY_RANDOM`` is an sktime-only backbone used for
deterministic CPU coverage of patch embedding, prompting, reprogramming,
and the projection head. ``get_test_params`` also uses ``TINY_RANDOM``
only; it does not load GPT2 or BERT.

``short_term_forecast`` and ``long_term_forecast`` share the same
``Model.forward`` branch. Layer-count coverage is therefore a separate
``llm_layers=2`` case, not bundled with ``task_name="short_term_forecast"``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.time_llm import TimeLLMForecaster
from sktime.tests.test_switch import run_test_for_class

_RTOL = 1e-5
_ATOL = 1e-4

_TINY_RANDOM_BASE = {
    "pred_len": 12,
    "llm_model": "TINY_RANDOM",
    "llm_dim": 1,
    "d_model": 8,
    "d_ff": 1,
    "n_heads": 1,
    "dropout": 0.0,
    "prompt_domain": False,
}

_TIME_LLM_PARITY_CASES = [
    pytest.param(
        {
            "task_name": "long_term_forecast",
            "llm_layers": 1,
            "patch_len": 16,
            "stride": 8,
        },
        id="tiny-random-long-l1-p16-s8",
    ),
    pytest.param(
        {
            "task_name": "long_term_forecast",
            "llm_layers": 1,
            "patch_len": 12,
            "stride": 6,
        },
        id="tiny-random-long-l1-p12-s6",
    ),
    pytest.param(
        {
            "task_name": "long_term_forecast",
            "llm_layers": 2,
            "patch_len": 16,
            "stride": 8,
        },
        id="tiny-random-long-l2-p16-s8",
    ),
]


pytestmark = pytest.mark.skipif(
    not run_test_for_class(TimeLLMForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _airline_split():
    y = load_airline()
    y_train = y.iloc[:-12]
    fh = np.arange(1, 13)
    return y_train, fh


def _encode_tensor(y_train, torch):
    """Build the same CPU float32 encoding as ``TimeLLMForecaster._predict``."""
    values = y_train.to_frame().values if y_train.ndim == 1 else y_train.values
    x = torch.tensor(values).reshape(1, -1, 1).to("cpu")
    return x.to(torch.float32)


def _head(values):
    return np.asarray(values, dtype=np.float64).ravel()[:3]


def _assert_head_close(actual, expected):
    np.testing.assert_allclose(_head(actual), _head(expected), rtol=_RTOL, atol=_ATOL)


def _source_heads(y_train, n_passes=1, **model_kwargs):
    """Run vendored Time-LLM Model.forward, bypassing TimeLLMForecaster."""
    import torch

    from sktime.libs.time_llm.TimeLLM import Model

    cfg = SimpleNamespace(enc_in=1, **model_kwargs)
    torch.manual_seed(0)
    model = Model(cfg).to("cpu").to(torch.bfloat16).eval()
    x = _encode_tensor(y_train, torch)
    heads = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model.forward(x, x_mark_enc=None, x_mark_dec=None, x_dec=None)
            heads.append(out.detach().float().cpu().numpy().ravel())
    return heads


def _wrapper_head(y_train, fh, **forecaster_kwargs):
    import torch

    forecaster = TimeLLMForecaster(device="cpu", **forecaster_kwargs)
    torch.manual_seed(0)
    return forecaster.fit(y_train, fh=fh).predict(fh=fh).to_numpy().ravel()


def _tiny_random_kwargs(**overrides):
    y_train, fh = _airline_split()
    kwargs = {
        **_TINY_RANDOM_BASE,
        "seq_len": len(y_train),
        **overrides,
    }
    return y_train, fh, kwargs


@pytest.mark.parametrize("forecaster_kwargs", _TIME_LLM_PARITY_CASES)
def test_time_llm_predict_matches_vendored_model_forward(forecaster_kwargs):
    """Wrapper predict matches same-process vendored Model.forward.

    Call-path parity only: both sides use the sktime fork of Time-LLM,
    not a pinned KimMeen/Time-LLM checkout.
    """
    y_train, fh, kwargs = _tiny_random_kwargs(**forecaster_kwargs)
    source = _source_heads(y_train, n_passes=1, **kwargs)[0]
    wrapper = _wrapper_head(y_train, fh, **kwargs)
    _assert_head_close(wrapper, source)


def test_time_llm_short_term_is_forward_alias_of_long_term():
    """short_term_forecast shares Model.forward with long_term_forecast."""
    y_train, fh, long_kwargs = _tiny_random_kwargs(
        task_name="long_term_forecast",
        llm_layers=1,
        patch_len=16,
        stride=8,
    )
    short_kwargs = {**long_kwargs, "task_name": "short_term_forecast"}

    long_source = _source_heads(y_train, **long_kwargs)[0]
    short_source = _source_heads(y_train, **short_kwargs)[0]
    np.testing.assert_allclose(short_source, long_source, rtol=_RTOL, atol=_ATOL)

    long_wrapper = _wrapper_head(y_train, fh, **long_kwargs)
    short_wrapper = _wrapper_head(y_train, fh, **short_kwargs)
    np.testing.assert_allclose(short_wrapper, long_wrapper, rtol=_RTOL, atol=_ATOL)
    _assert_head_close(short_wrapper, short_source)


def test_time_llm_gpt2_predict_matches_vendored_model_forward():
    """GPT2 backbone call-path parity, skipped if forward is unstable.

    GPT2 is the default Time-LLM LLM in KimMeen/Time-LLM. On CPU bfloat16
    the SDPA path is often not self-consistent across two Model.forward
    calls, so this test skips instead of freezing noisy literals.
    """
    y_train, fh = _airline_split()
    kwargs = {
        "task_name": "long_term_forecast",
        "pred_len": 12,
        "seq_len": len(y_train),
        "llm_model": "GPT2",
        "llm_layers": 3,
        "llm_dim": 768,
        "patch_len": 16,
        "stride": 8,
        "d_model": 128,
        "d_ff": 128,
        "n_heads": 4,
        "dropout": 0.0,
        "prompt_domain": False,
    }
    try:
        source_a, source_b = _source_heads(y_train, n_passes=2, **kwargs)
    except (OSError, RuntimeError, ValueError, ImportError) as err:
        pytest.skip(
            "GPT2 backbone could not be loaded for call-path check: "
            f"{type(err).__name__}: {err}"
        )
    if not np.allclose(source_a, source_b, rtol=_RTOL, atol=_ATOL):
        pytest.skip(
            "GPT2 CPU bfloat16 Model.forward is not self-consistent (SDPA); "
            "skipping rather than locking a noisy literal"
        )
    wrapper = _wrapper_head(y_train, fh, **kwargs)
    _assert_head_close(wrapper, source_a)
