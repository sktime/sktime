# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""patched_decoder."""

import dataclasses

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("einshape", severity="none"):
    import einshape as es

if _check_soft_dependencies("jax", severity="none"):
    import jax.numpy as jnp
    from jax import lax

if _check_soft_dependencies("praxis", severity="none"):
    from praxis import base_layer, layers, pax_fiddle, py_utils
    from praxis.base_layer import BaseLayer
    from praxis.base_model import BaseModel
    from praxis.layers import (
        activations,
        embedding_softmax,
        linears,
        normalizations,
        stochastics,
        transformers,
    )

    # PAX shortcuts
    NestedMap = py_utils.NestedMap

    LayerTpl = pax_fiddle.Config[BaseLayer]
    template_field = base_layer.template_field

else:

    class BaseLayer:
        """Dummy class if praxis is unavailable."""

    class BaseModel:
        """Dummy class if praxis is unavailable."""


PAD_VAL = 1123581321.0
DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# NestedMap keys
_INPUT_TS = "input_ts"
_TARGET_FUTURE = "actual_ts"
_INPUT_PADDING = "input_padding"
_OUTPUT_TS = "output_ts"
_FREQ = "freq"
_OUTPUT_TOKENS = "output_tokens"
_STATS = "stats"


# Small numerical value.
_TOLERANCE = 1e-7


def _shift_padded_seq(mask, seq):
    num = seq.shape[1]

    # Find the index of the first 0 in each row of the mask
    first_zero_idx = jnp.argmin(mask, axis=1)

    # Create a range array for indexing
    idx_range = jnp.arange(num)

    def shift_row(carry, x):
        seq_row, shift = x
        shifted_idx = (idx_range - shift) % num
        shifted_row = seq_row[shifted_idx]
        return carry, shifted_row

    # Use lax.scan to shift each row of seq based on the corresponding
    # first_zero_idx.
    _, shifted_seq = lax.scan(shift_row, None, (seq, first_zero_idx))

    return shifted_seq


class ResidualBlock(BaseLayer):
    """ResidualBlock."""

    if _check_soft_dependencies("praxis", severity="none"):
        # These should not be changed or removed
        input_dims: int = 0
        hidden_dims: int = 0
        output_dims: int = 0
        dropout_prob: float = 0.0
        layer_norm: bool = False
        dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
        ln_tpl: LayerTpl = template_field(normalizations.LayerNorm)
        act_tpl: LayerTpl = template_field(activations.Swish)

    def setup(self):
        """Set up."""
        lnorm_tpl = self.ln_tpl.clone()
        lnorm_tpl.dim = self.output_dims
        self.create_child("ln_layer", lnorm_tpl)

        dropout_tpl = self.dropout_tpl.clone()
        dropout_tpl.keep_prob = 1.0 - self.dropout_prob
        self.create_child("dropout", dropout_tpl)

        self.create_child(
            "hidden_layer",
            pax_fiddle.Config(
                linears.FeedForward,
                input_dims=self.input_dims,
                output_dims=self.hidden_dims,
                activation_tpl=self.act_tpl.clone(),
            ),
        )

        self.create_child(
            "output_layer",
            pax_fiddle.Config(
                linears.FeedForward,
                input_dims=self.hidden_dims,
                output_dims=self.output_dims,
                activation_tpl=pax_fiddle.Config(activations.Identity),
            ),
        )

        self.create_child(
            "residual_layer",
            pax_fiddle.Config(
                linears.FeedForward,
                input_dims=self.input_dims,
                output_dims=self.output_dims,
                activation_tpl=pax_fiddle.Config(activations.Identity),
            ),
        )

    def __call__(self, inputs):
        """__call__."""
        hidden = self.hidden_layer(inputs)
        output = self.output_layer(hidden)
        output = self.dropout(output)
        residual = self.residual_layer(inputs)
        if self.layer_norm:
            return self.ln_layer(output + residual)
        else:
            return output + residual


def _masked_mean_std(inputs, padding):
    pad_sum = jnp.sum(1 - padding, axis=2)

    def _get_patch_index(arr):
        indices = jnp.argmax(arr >= 3, axis=1)
        row_sum = (arr >= 3).sum(axis=1)
        return jnp.where(row_sum == 0, arr.shape[1] - 1, indices)

    patch_indices = _get_patch_index(pad_sum)
    bidxs = jnp.arange(inputs.shape[0])

    arr = inputs[bidxs, patch_indices, :]
    pad = padding[bidxs, patch_indices, :]

    # Create a mask where P is 0
    mask = 1 - pad

    # Calculate the number of valid elements
    num_valid_elements = jnp.sum(mask, axis=1)

    num_valid_elements = jnp.where(num_valid_elements == 0, 1, num_valid_elements)

    # Calculate the masked sum and squared sum of M
    masked_sum = jnp.sum(arr * mask, axis=1)
    masked_squared_sum = jnp.sum((arr * mask) ** 2, axis=1)

    # Calculate the masked mean and standard deviation
    masked_mean = masked_sum / num_valid_elements
    masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
    masked_var = jnp.where(masked_var < 0.0, 0.0, masked_var)
    masked_std = jnp.sqrt(masked_var)

    return masked_mean, masked_std


def _create_quantiles():
    return DEFAULT_QUANTILES


class PatchedTimeSeriesDecoder(BaseLayer):
    """PatchedTimeSeriesDecoder."""

    if _check_soft_dependencies("praxis", severity="none"):
        # These should not be changed or removed
        patch_len: int = 0
        horizon_len: int = 0
        model_dims: int = 0
        hidden_dims: int = 0
        quantiles: list[float] = dataclasses.field(default_factory=_create_quantiles)
        residual_block_tpl: LayerTpl = template_field(ResidualBlock)
        stacked_transformer_params_tpl: LayerTpl = template_field(
            transformers.StackedTransformer
        )
        use_freq: bool = True

    def setup(self):
        """Construct the model."""
        num_outputs = len(self.quantiles) + 1

        stl = self.stacked_transformer_params_tpl.clone()
        stl.model_dims = self.model_dims
        stl.hidden_dims = self.hidden_dims
        stl.mask_self_attention = True

        self.create_child("stacked_transformer_layer", stl)

        input_resl = self.residual_block_tpl.clone()
        ff_in_dims = 2 * self.patch_len
        input_resl.input_dims = ff_in_dims
        input_resl.hidden_dims = self.hidden_dims
        input_resl.output_dims = self.model_dims
        self.create_child(
            "input_ff_layer",
            input_resl,
        )

        horizon_resl = self.residual_block_tpl.clone()
        horizon_resl.input_dims = self.model_dims
        horizon_resl.hidden_dims = self.hidden_dims
        horizon_resl.output_dims = self.horizon_len * num_outputs
        self.create_child(
            "horizon_ff_layer",
            horizon_resl,
        )

        self.create_child(
            "position_emb",
            pax_fiddle.Config(
                layers.PositionalEmbedding, embedding_dims=self.model_dims
            ),
        )

        if self.use_freq:
            self.create_child(
                "freq_emb",
                pax_fiddle.Config(
                    embedding_softmax.Embedding,
                    num_classes=3,
                    input_dims=self.model_dims,
                ),
            )

    def transform_decode_state(self, transform_fn):
        """transform_decode_state."""
        self.stacked_transformer_layer.transform_decode_state(transform_fn)

    def _forward_transform(self, inputs, patched_pads):
        """Input is of shape [B, N, P]."""
        mu, sigma = _masked_mean_std(inputs, patched_pads)
        sigma = jnp.where(sigma < _TOLERANCE, 1.0, sigma)
        # Normalize each patch.
        outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
        outputs = jnp.where(jnp.abs(inputs - PAD_VAL) < _TOLERANCE, PAD_VAL, outputs)
        return outputs, (mu, sigma)

    def _reverse_transform(self, outputs, stats):
        """Output is of shape [B, N, P, Q]."""
        mu, sigma = stats
        return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

    def _preprocess_input(
        self,
        input_ts,
        input_padding,
        pos_emb=None,
    ):
        """Preprocess input for stacked transformer."""
        # Reshape into patches.
        patched_inputs = es.jax_einshape("b(np)->bnp", input_ts, p=self.patch_len)
        patched_pads = es.jax_einshape("b(np)->bnp", input_padding, p=self.patch_len)
        patched_inputs = jnp.where(
            jnp.abs(patched_pads - 1.0) < _TOLERANCE, 0.0, patched_inputs
        )
        patched_pads = jnp.where(
            jnp.abs(patched_inputs - PAD_VAL) < _TOLERANCE, 1, patched_pads
        )
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = jnp.concatenate([patched_inputs, patched_pads], axis=-1)
        model_input = self.input_ff_layer(concat_inputs)
        # A patch should not be padded even if there is at least one zero.
        patched_padding = jnp.min(patched_pads, axis=-1)

        if pos_emb is None:
            position_emb = self.position_emb(seq_length=model_input.shape[1])
        else:
            position_emb = pos_emb
        if self.do_eval:
            if position_emb.shape[0] != model_input.shape[0]:
                position_emb = jnp.repeat(position_emb, model_input.shape[0], axis=0)
            position_emb = _shift_padded_seq(patched_padding, position_emb)
        model_input += position_emb

        return model_input, patched_padding, stats, patched_inputs

    def _postprocess_output(
        self,
        model_output,
        num_outputs,
        stats,
    ):
        """Postprocess output of stacked transformer."""
        # B x N x (H.Q)
        output_ts = self.horizon_ff_layer(model_output)
        output_ts = es.jax_einshape(
            "bn(hq)->bnhq", output_ts, q=num_outputs, h=self.horizon_len
        )
        return self._reverse_transform(output_ts, stats)

    def __call__(self, inputs):
        """PatchTST call.

        Args:
          inputs: A NestedMap containing (1) input_ts: input sequence of shape [B,
            T] where T must be multiple of patch_length; (2) input_padding: that
            contains padding map.

        Returns
        -------
          A nested map with two keys:
          (1) 'output_tokens' of shape [B, N, D].
          (2) 'output_ts' of shape [B, N, H, Q]
          (3) 'stats' a tuple of statistics for renormalization.
        """
        input_ts, input_padding = inputs[_INPUT_TS], inputs[_INPUT_PADDING]
        num_outputs = len(self.quantiles) + 1
        model_input, patched_padding, stats, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )
        if self.use_freq:
            freq = inputs[_FREQ].astype(jnp.int32)
            f_emb = self.freq_emb(freq)  # B x 1 x D
            f_emb = jnp.repeat(f_emb, model_input.shape[1], axis=1)
            model_input += f_emb
        model_output = self.stacked_transformer_layer(model_input, patched_padding)

        output_ts = self._postprocess_output(model_output, num_outputs, stats)
        return NestedMap(
            {_OUTPUT_TOKENS: model_output, _OUTPUT_TS: output_ts, _STATS: stats}
        )

    def decode(
        self,
        inputs,
        horizon_len,
        output_patch_len=None,
        max_len=512,
        return_forecast_on_context=False,
    ):
        """Decode."""
        final_out = inputs[_INPUT_TS]
        context_len = final_out.shape[1]
        paddings = inputs[_INPUT_PADDING]
        if self.use_freq:
            freq = inputs[_FREQ].astype(jnp.int32)
        else:
            freq = jnp.zeros([final_out.shape[0], 1], dtype=jnp.int32)
        full_outputs = []
        if paddings.shape[1] != final_out.shape[1] + horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}"
            )
        if output_patch_len is None:
            output_patch_len = self.horizon_len
        num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len
        for step_index in range(num_decode_patches):
            current_padding = paddings[:, 0 : final_out.shape[1]]
            input_ts = final_out[:, -max_len:]
            input_padding = current_padding[:, -max_len:]
            model_input = NestedMap(
                input_ts=input_ts,
                input_padding=input_padding,
                freq=freq,
            )
            fprop_outputs = self(model_input)[_OUTPUT_TS]
            if return_forecast_on_context and step_index == 0:
                # For the first decodings step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[:, :-1, : self.patch_len, :]
                new_full_ts = es.jax_einshape("bnph->b(np)h", new_full_ts)

                full_outputs.append(new_full_ts)

            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = jnp.concatenate([final_out, new_ts], axis=-1)

        if return_forecast_on_context:
            # `full_outputs` indexing starts at after the first input patch.
            full_outputs = jnp.concatenate(full_outputs, axis=1)[
                :, : (context_len - self.patch_len + horizon_len), :
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = jnp.concatenate(full_outputs, axis=1)[:, 0:horizon_len, :]

        return (full_outputs[:, :, 0], full_outputs)


class PatchedDecoderFinetuneModel(BaseModel):
    """PatchedDecoderFinetuneModel."""

    def __init__(self):
        self.core_layer_tpl = template_field(PatchedTimeSeriesDecoder)
        self.freq: int = 0

    def setup(self):
        """Set up."""
        self.create_child("core_layer", self.core_layer_tpl)

    def compute_predictions(self, input_batch):
        """compute_predictions."""
        input_ts = input_batch[_INPUT_TS]
        input_padding = jnp.zeros_like(input_ts)
        context_len = input_ts.shape[1]
        input_patch_len = self.core_layer_tpl.patch_len
        context_pad = (
            (context_len + input_patch_len - 1) // input_patch_len
        ) * input_patch_len - context_len

        input_ts = jnp.pad(input_ts, [(0, 0), (context_pad, 0)])
        input_padding = jnp.pad(
            input_padding, [(0, 0), (context_pad, 0)], constant_values=1
        )
        freq = jnp.ones([input_ts.shape[0], 1], dtype=jnp.int32) * self.freq
        new_input_batch = NestedMap(
            input_ts=input_ts,
            input_padding=input_padding,
            freq=freq,
        )
        return self.core_layer(new_input_batch)

    def _quantile_loss(self, pred, actual, quantile: float):
        dev = actual - pred
        loss_first = dev * quantile
        loss_second = -dev * (1.0 - quantile)
        return 2 * jnp.where(loss_first >= 0, loss_first, loss_second)

    def compute_loss(self, prediction_output, input_batch):
        """compute_loss."""
        output_ts = prediction_output[_OUTPUT_TS]
        actual_ts = input_batch[_TARGET_FUTURE]
        pred_ts = output_ts[:, -1, 0 : actual_ts.shape[1], :]
        loss = jnp.square(pred_ts[:, :, 0] - actual_ts)
        for i, quantile in enumerate(self.core_layer.quantiles):
            loss += self._quantile_loss(pred_ts[:, :, i + 1], actual_ts, quantile)
        loss = loss.mean()
        loss_weight = jnp.array(1.0, dtype=jnp.float32)
        per_example_out = NestedMap()
        return {"avg_qloss": (loss, loss_weight)}, per_example_out
