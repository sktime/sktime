"""Tests for TimesFMForecaster.

TimesFmForecaster has a very restrictive dependency set,
which may cancel out with the matrix testing strategy.

Therefore, we call a check_estimator separately.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timesfm import TimesFMForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils import check_estimator

_TIMESFM_JAX_REFERENCE_CASES = [
    pytest.param(
        128,
        [1, 2, 3],
        [401.7045593261719, 392.53509521484375, 429.7332458496094],
        id="horizon_len-128",
    ),
    pytest.param(
        256,
        [254, 255, 256],
        [586.6953735351562, 622.218505859375, 620.0606079101562],
        id="horizon_len-256",
    ),
]


@pytest.mark.skipif(
    not run_test_for_class(TimesFMForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timesfmforecaster():
    """Run standard test suite for TimesFMForecaster.

    TimesFmForecaster has a very restrictive dependency set,
    which may cancel out with the matrix testing strategy.

    Therefore, we call a check_estimator separately.
    """
    check_estimator(TimesFMForecaster, raise_exceptions=True)


@pytest.mark.skipif(
    not run_test_for_class(TimesFMForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("horizon_len, fh, expected", _TIMESFM_JAX_REFERENCE_CASES)
def test_timesfm_airline_predictions_match_jax_reference(horizon_len, fh, expected):
    """TimesFM predictions match those of the previous JAX implementation.

    The expected forecasts were computed with the JAX implementation of TimesFM
    previously used by ``TimesFMForecaster``, ``sktime.libs.timesfm``, with the
    ``google/timesfm-1.0-200m`` checkpoint, the default ``context_len``, 160, and
    the given ``horizon_len``, on ``load_airline().iloc[:-12]``. Forecasts beyond
    step 128, the output patch length, are obtained by autoregressive decoding.
    """
    y_train = load_airline().iloc[:-12]

    forecaster = TimesFMForecaster(horizon_len=horizon_len)
    y_pred = forecaster.fit(y_train, fh=fh).predict()

    np.testing.assert_allclose(
        y_pred.to_numpy(),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )


def _timesfm_checkpoint(num_layers, model_dims, patch_len, horizon_len):
    """Return random weights, named and shaped as in TimesFM 1.0 checkpoints.

    Names and shapes are those of ``torch_model.ckpt`` in
    ``google/timesfm-1.0-200m-pytorch``, for the given architecture.
    """
    import torch

    output_dims = horizon_len * 10
    shapes = {
        "input_ff_layer.hidden_layer.0.weight": (model_dims, 2 * patch_len),
        "input_ff_layer.hidden_layer.0.bias": (model_dims,),
        "input_ff_layer.output_layer.weight": (model_dims, model_dims),
        "input_ff_layer.output_layer.bias": (model_dims,),
        "input_ff_layer.residual_layer.weight": (model_dims, 2 * patch_len),
        "input_ff_layer.residual_layer.bias": (model_dims,),
        "freq_emb.weight": (3, model_dims),
        "horizon_ff_layer.hidden_layer.0.weight": (model_dims, model_dims),
        "horizon_ff_layer.hidden_layer.0.bias": (model_dims,),
        "horizon_ff_layer.output_layer.weight": (output_dims, model_dims),
        "horizon_ff_layer.output_layer.bias": (output_dims,),
        "horizon_ff_layer.residual_layer.weight": (output_dims, model_dims),
        "horizon_ff_layer.residual_layer.bias": (output_dims,),
    }
    for i in range(num_layers):
        layer = f"stacked_transformer.layers.{i}"
        shapes.update(
            {
                f"{layer}.self_attn.scaling": (model_dims // 16,),
                f"{layer}.self_attn.qkv_proj.weight": (3 * model_dims, model_dims),
                f"{layer}.self_attn.qkv_proj.bias": (3 * model_dims,),
                f"{layer}.self_attn.o_proj.weight": (model_dims, model_dims),
                f"{layer}.self_attn.o_proj.bias": (model_dims,),
                f"{layer}.mlp.gate_proj.weight": (model_dims, model_dims),
                f"{layer}.mlp.gate_proj.bias": (model_dims,),
                f"{layer}.mlp.down_proj.weight": (model_dims, model_dims),
                f"{layer}.mlp.down_proj.bias": (model_dims,),
                f"{layer}.mlp.layer_norm.weight": (model_dims,),
                f"{layer}.mlp.layer_norm.bias": (model_dims,),
                f"{layer}.input_layernorm.weight": (model_dims,),
            }
        )
    return {name: torch.randn(shape) for name, shape in shapes.items()}


@pytest.mark.skipif(
    not run_test_for_class(TimesFMForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timesfm_checkpoint_conversion():
    """Weights of TimesFM 1.0 PyTorch checkpoints load into transformers TimesFM."""
    import torch

    from sktime.forecasting.timesfm import _convert_timesfm_state_dict, _TimesFMTorch

    model_dims = 16
    checkpoint = _timesfm_checkpoint(
        num_layers=2, model_dims=model_dims, patch_len=32, horizon_len=32
    )
    tfm = _TimesFMTorch(
        context_len=64,
        horizon_len=48,
        input_patch_len=32,
        output_patch_len=32,
        num_layers=2,
        model_dims=model_dims,
        per_core_batch_size=1,
        backend="cpu",
        verbose=False,
    )
    tfm.load_state_dict(checkpoint)
    weights = tfm.model.state_dict()

    for name, value in _convert_timesfm_state_dict(checkpoint).items():
        assert torch.equal(weights[name], value)

    for i in range(2):
        qkv = checkpoint[f"stacked_transformer.layers.{i}.self_attn.qkv_proj.weight"]
        for j, proj in enumerate(["q_proj", "k_proj", "v_proj"]):
            assert torch.equal(
                weights[f"decoder.layers.{i}.self_attn.{proj}.weight"],
                qkv[j * model_dims : (j + 1) * model_dims],
            )

    mean, full = tfm.forecast([np.arange(50.0), -np.arange(70.0)])
    assert mean.shape == (2, 48)
    assert full.shape == (2, 48, 10)
    np.testing.assert_array_equal(mean, full[:, :, 0])

    del checkpoint["freq_emb.weight"]
    with pytest.raises(ValueError, match="does not match the model architecture"):
        tfm.load_state_dict(checkpoint)
