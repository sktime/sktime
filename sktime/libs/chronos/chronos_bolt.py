"""Chronos-Bolt: Improved version of Chronos for time-series forecasting.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Authors: Abdul Fatir Ansari <ansarnd@amazon.com>, Caner Turkmen <atturkm@amazon.com>, Lorenzo Stella <stellalo@amazon.com>
Original source:
https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/src/autogluon/timeseries/models/chronos/pipeline/chronos_bolt.py
"""  # noqa: E501

import copy
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Union

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
else:

    class torch:
        """Dummy class if torch is unavailable."""

        class Tensor:
            """Dummy class if torch is unavailable."""

    class nn:
        """Dummy class if torch is unavailable."""

        class Module:
            """Dummy class if torch is unavailable."""


if _check_soft_dependencies("transformers", severity="none"):
    from transformers import AutoConfig
    from transformers.models.t5.modeling_t5 import (
        ACT2FN,
        T5LayerNorm,
        T5PreTrainedModel,
        T5Stack,
    )
    from transformers.utils import ModelOutput
else:

    class T5PreTrainedModel:
        """Dummy class if transformers is unavailable."""

    class ModelOutput:
        """Dummy model output if transformers is unavailable."""


logger = logging.getLogger(__file__)


@dataclass
class ChronosBoltConfig:
    """Configuration for chronos-bolt."""

    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: list[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    """Description of the output of the model."""

    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    """
    Divides an input tensor into patches.

    Parameters
    ----------
        patch_size (int): The size of each patch.
        patch_stride (int): The stride between patches.
    """

    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input time series `x` into patches for further processing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length).

        Returns
        -------
        x_with_patches: torch.Tensor
            Tensor of shape (batch_size, num_patches, patch_size).
        """
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """See, also, RevIN. Apply standardization along the last dimension."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply instance normalization to the input tensor.

        This method normalizes the input tensor by subtracting the mean and dividing by
        the standard deviation, computed separately for each instance of the batch.

        Parameter
        ---------
        x: torch.Tensor
            Input tensor of the shape (batch_size, channels, height, width) or
            (batch_size, features).
        loc_scale: tuple[torch.Tensor, torch.Tensor], optional (default=None)
            Represent the location (mean) and the scale (standard deviation) for
            normalization. If not provided explicitly, the function calculates this
            from the data in ``x``.
        """
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, torch.abs(loc) + self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Reverses the normalization process of the InstanceNorm during ``forward()``.

        Parameter
        ---------
        x: torch.Tensor
            The normalised tensor, typically output from the `forward` method.
        loc_scale: tuple[torch.Tensor, torch.Tensor]
            Tuple containing the `loc` and `scale` values for the inverse function.

        Returns
        -------
        x: torch.tensor
            The original unnormalized tensor.
        """
        loc, scale = loc_scale
        return x * scale + loc


class ResidualBlock(nn.Module):
    """
    A Residual Block for optional LayerNorm and Dropout.

    This block implementes a standard residual connection that allows gradients to flow
    through the layers more effectively. It consists of two layers with an activation
    function in between with an option dropout and layer normalization operation.

    Parameters
    ----------
    in_dim : int
        The number of input features.
    h_dim : int
        The number of hidden units in the hidden layer.
    out_dim : int
        The number of output features from this block.
    act_fn_name : str
        The name of the activation function to use (e.g., 'relu', 'tanh').
    dropout_p : float, optional (default = 0.0)
        The dropout probability for regularization (default is 0.0).
    use_layer_norm : bool, optional (default = False)
        If True, applies Layer Normalization after adding the residual.
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the Residual Block.

        Parameter
        ---------
        x: torch.Tensor
            Input tensor with shape (batch_size, in_dim)

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, out_dim), which is computed as the sum
            of the output from this block and its corresponding residual connections.
            If LayerNorm is used it will applied before returning.
        """
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class ChronosBoltModelForForecasting(T5PreTrainedModel):
    """
    Wraps a `T5PreTrainedModel` object from `transformers`.

    Parameters
    ----------
    config: ChronosBoltConfig
        The configuration to use for ChronosBolt models.
    """

    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: ChronosBoltConfig):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)
        self.model_dim = config.d_model

        self.chronos_config = ChronosBoltConfig(**config.chronos_config)

        self.config.context_length = self.chronos_config.context_length
        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self._init_decoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.hidden_layer, "bias")
                and module.hidden_layer.bias is not None
            ):
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.residual_layer, "bias")
                and module.residual_layer.bias is not None
            ):
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if (
                hasattr(module.output_layer, "bias")
                and module.output_layer.bias is not None
            ):
                module.output_layer.bias.data.zero_()

    def encode(
        self, context: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """
        Encode the input context tensor using the model's architecture.

        Parameters
        ----------
            context (torch.Tensor):
                Input tensor to be encoded, shape (batch_size, seq_length)
            mask (Optional[torch.Tensor]):
                Optional mask tensor with same shape as context.
                If None, mask is created based on NaN values in context.

        Returns
        -------
            tuple containing:
                - encoder_outputs[0]: torch.Tensor
                    Encoded representation from the encoder.
                - loc_scale: tuple[torch.Tensor, torch.Tensor]
                    Location and scale parameters from instance norm.
                - input_embeds: torch.Tensor
                    Input embeddings after patch embedding.
                - attention_mask: torch.Tensor:
                    Attention mask used in the encoder.
        """
        mask = (
            mask.to(context.dtype)
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        # scaling
        context, loc_scale = self.instance_norm(context)

        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = (
            patched_mask.sum(dim=-1) > 0
        )  # (batch_size, patched_seq_length)

        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )

        return encoder_outputs[0], loc_scale, input_embeds, attention_mask

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        """
        Predict the future sample tokens for the given sequences.

        Arguments `context`, `mask`, `target` and `target_mask` can be used to customize
        the model inference at runtime.

        Parameters
        ----------
        context: torch.Tensor
            Input time series of the shape (batch_size, sequence_length)
        mask: torch.Tensor, optional (default=None)
            A mask tensor of the same shape as context to avoid attending
            on padding or missing tokens.
        target: torch.Tensor, optional (default=None)
            Actual values of the target time series that the model is trying to predict.
        target_mask: torch.Tensor, optional (default=None)
            The mask tensor of the same shape as target to avoid attending on padding or
            missing tokens.

        Returns
        -------
        output_description: tuple
            Contains description of the model outputl. Refer `ChronosBoltOutput` for
            more context.
        """
        batch_size = context.size(0)

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        loss = None
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's pred_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)], dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=quantile_preds,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """Decode the inputs using the decoder stack to get the last hidden state.

        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs.
            Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context.
            Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder.
            Shape (batch_size, patched_context_length, d_model)
        output_attentions: bool, default=False
            Whether to return attention weights

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state  # sequence_outputs, b x 1 x d_model


def left_pad_and_stack_1D(tensors: list):
    """Pad a list of 1D tensors with ``torch.nan`` and stack them."""
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


@dataclass
class ChronosBoltPipeline:
    """
    Uses the given tokeniser and model to forecast the input time series.

    Use the ``from_trained()`` class method to load the serialized models.
    Use the ``predict`` method to get the forecasts.

    Parameters
    ----------
    model: ChronosBoltModelForForecasting
        The model to use.

    default_context_length: int (default=2048)
        The default context length of the model in the case that it is not explicitly
        specified.
    """

    model: ChronosBoltModelForForecasting
    default_context_length: int = 2048

    def _prepare_and_validate_context(
        self, context: Union[torch.Tensor, list[torch.Tensor]]
    ):
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    def embed(
        self, context: Union[torch.Tensor, list[torch.Tensor]]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Get encoder embeddings for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.

        Returns
        -------
        embeddings, loc_scale
            A tuple of two items: the encoder embeddings and the loc_scale,
            i.e., the mean and std of the original time series.
            The encoder embeddings are shaped (batch_size, num_patches + 1, d_model),
            where num_patches is the number of patches in the time series
            and the extra 1 is for the [REG] token (if used by the model).
        """
        with torch.no_grad():
            context_tensor = self._prepare_and_validate_context(context=context)
            model_context_length = self.model.config.chronos_config["context_length"]

            if context_tensor.shape[-1] > model_context_length:
                context_tensor = context_tensor[..., -model_context_length:]

            context_tensor = context_tensor.to(
                device=self.model.device,
                dtype=torch.float32,
            )
            embeddings, loc_scale, *_ = self.model.encode(context=context_tensor)
            return embeddings.cpu(), (
                loc_scale[0].squeeze(-1).cpu(),
                loc_scale[1].squeeze(-1).cpu(),
            )

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, list[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        """
        Get forecasts for the given time series.

        Refer to the base method (``BaseChronosPipeline.predict``)
        for details on shared parameters.
        Additional parameters
        ---------------------
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. False by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        torch.Tensor
            Forecasts of shape (batch_size, num_quantiles, prediction_length)
            where num_quantiles is the number of quantiles the model has been
            trained to output. For official Chronos-Bolt models, the value of
            num_quantiles is 9 for [0.1, 0.2, ..., 0.9]-quantiles.

        Raises
        ------
        ValueError
            When limit_prediction_length is True and the prediction_length is
            greater than model's trainig prediction_length.
        """
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "  # noqa: E501
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."  # noqa: E501
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance
        # collapsing every 64 steps.
        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor,
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.

        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers.
        """
        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting"  # noqa: E501
            )
            class_ = ChronosBoltModelForForecasting

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)
