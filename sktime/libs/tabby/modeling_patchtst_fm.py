# Copyright contributors to the TSFM project
#
# ruff: noqa
"""PatchTST-FM model implementation"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")
F = _safe_import("torch.nn.functional")
PreTrainedModel = _safe_import("transformers.modeling_utils.PreTrainedModel")
ModelOutput = _safe_import("transformers.utils.ModelOutput")
logging = _safe_import("transformers.utils.logging")

if ModelOutput.__class__.__name__ == "CommonMagicMeta":

    class ModelOutput:
        """Dummy model output if transformers is unavailable."""


from .basic import (
    TransformerBlock,
    make_attn_mask,
)
from .configuration_patchtst_fm import PatchTSTFMConfig
from .normalization import RevIN
from .tools import count_parameters


logger = logging.get_logger(__name__)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, type="add"):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.type = type

    def forward(self, x):
        positions = torch.arange(x.size(-2), device=x.device).unsqueeze(0)
        pe = self.embedding(positions)
        if x.ndim == 4:
            pe = pe.unsqueeze(1)
        if self.type == "add":
            return x + pe
        elif self.type == "mul":
            return x * pe
        else:
            raise ValueError(f"Invalid type: {self.type}")


class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()

        self.layer1 = nn.Linear(d_in, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_out)
        self.residual = nn.Linear(d_in, d_out)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x))) + self.residual(x)


class PatchTSTFMPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = PatchTSTFMConfig
    base_model_prefix = "model"
    main_input_name = "inputs"
    supports_gradient_checkpointing = False


@dataclass
class PatchTSTFMModelOutput(ModelOutput):
    loss_mask: Optional[torch.Tensor] = None
    normed_target: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    quantile_outputs: Optional[torch.FloatTensor] = None


@dataclass
class PatchTSTFMPretrainingOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    quantile_outputs: Optional[torch.Tensor] = None


@dataclass
class PatchTSTFMPredictionOutput(ModelOutput):
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    prediction_outputs: Optional[torch.Tensor | List[torch.Tensor]] = None
    quantile_outputs: Optional[torch.Tensor | List[torch.Tensor]] = None


class PatchTSTFMModel(PatchTSTFMPreTrainedModel):
    def __init__(self, config: PatchTSTFMConfig):
        super().__init__(config)
        self.config = config
        self.quantile_levels = config.quantile_levels
        self.pos_embed = LearnedPositionalEmbedding(
            d_model=config.d_model, max_len=config.n_patch, type="add"
        )
        assert config.d_model % config.n_head == 0, (
            "[QuantileDecoder] d_model must be divisible by n_head"
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_head,
                    mlp_ratio=4.0,
                    norm_first=config.norm_first,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.in_layer = ResidualBlock(
            config.d_patch * 2, config.d_model, config.d_model
        )
        self.out_layer = ResidualBlock(
            config.d_model, config.d_patch * (config.num_quantile + 1), config.d_model
        )

        self.norm_fn = RevIN(dim=-1, std_min=1e-5, use_sinh=True)

    def model_summary(self):
        s = ""
        model_name = "PatchTST-FM"
        s += f"{'=' * 5:<10} {model_name} {'=' * 5:>9}\n"
        s += f"{'Transformer:':<20} {count_parameters(self.blocks)[0] / 1e6:>8.2f}M\n"
        s += f"{'=' * 30}\n"
        p = count_parameters(self)
        s += f"{'Trainable:':<20} {p[1] / 1e6:>8.2f}M\n"
        s += f"{'Frozen:':<20} {p[2] / 1e6:>8.2f}M\n"
        s += f"{'Total:':<20} {p[0] / 1e6:>8.2f}M\n"
        s += f"{'=' * 30}\n"
        return s

    def forward(
        self,
        inputs: torch.Tensor,
        pred_mask: torch.Tensor,
        miss_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        # **kwargs,
    ) -> PatchTSTFMModelOutput:
        x = inputs  # .to(self.device)
        pad_mask = pad_mask.bool()  # to(self.device).bool()
        pred_mask = pred_mask.bool()  # to(self.device).bool()
        miss_mask = miss_mask.bool()  # to(self.device).bool()
        if x.ndim > 2:
            batch_size, num_channels, time_points = x.shape
            x = x.reshape(batch_size * num_channels, time_points)
            pad_mask = pad_mask.reshape(batch_size * num_channels, time_points)
            pred_mask = pred_mask.reshape(batch_size * num_channels, time_points)
            miss_mask = miss_mask.reshape(batch_size * num_channels, time_points)

        B, T = x.shape
        ts_mask = pred_mask | pad_mask | miss_mask

        x_target = self.norm_fn.fit_transform(x, mask=pred_mask | pad_mask | miss_mask)
        x_input = torch.where(ts_mask, torch.zeros_like(x_target), x_target)

        x_patch = x_input.reshape(B, self.config.n_patch, self.config.d_patch)
        mask_patch = ts_mask.reshape(B, self.config.n_patch, self.config.d_patch)
        pad_patch_mask = (
            pad_mask.reshape(B, self.config.n_patch, self.config.d_patch)
            .float()
            .mean(dim=-1)
            .gt(0.9)
        )

        q_pred, q_raw = self.decode(
            x=x_patch, mask=mask_patch.float(), t_pad_mask=pad_patch_mask
        )
        q_pred = q_pred.permute(0, 2, 3, 1)

        B, N, D, Q = q_pred.shape
        q_pred = q_pred.reshape(B, N * D, Q)

        if output_hidden_states:
            hidden_states = q_raw.reshape(B, N * D, Q)
        else:
            hidden_states = None

        # return here q_pred, loss_mask, and x_target
        return PatchTSTFMModelOutput(
            normed_target=x_target,
            quantile_outputs=q_pred,
            loss_mask=(pred_mask & ~pad_mask & ~miss_mask).float(),
            hidden_states=hidden_states,
        )

    def decode(self, x, mask, t_pad_mask=None):
        B, N, D = x.shape
        # x = self.in_layer(torch.cat([x, t, 1 - mask], dim=-1))
        x = self.in_layer(torch.cat([x, 1 - mask], dim=-1))
        pad_attn_mask = make_attn_mask(t_pad_mask, t_pad_mask).unsqueeze(1)

        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, pad_attn_mask)
        x = self.out_layer(x)
        q_raw = x.reshape(
            B, N, self.config.num_quantile + 1, self.config.d_patch
        ).permute(0, 2, 1, 3)
        q = q_raw[:, 0, :, :].unsqueeze(1) + torch.cumsum(
            F.softplus(q_raw[:, 1:, :, :]) / self.config.num_quantile, dim=1
        )
        return q, q_raw


class PatchTSTFMForPretraining(PatchTSTFMPreTrainedModel):
    def __init__(self, config: PatchTSTFMConfig):
        super().__init__(config)

        self.config = config
        self.backbone = PatchTSTFMModel(config)

        # move all out_layer items here

    def forward(
        self,
        inputs: torch.Tensor,
        pred_mask: torch.Tensor,
        miss_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> PatchTSTFMPretrainingOutput:
        # move quantile logic here

        model_outputs = self.backbone(
            inputs,
            pred_mask=pred_mask,
            miss_mask=miss_mask,
            pad_mask=pad_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        q_pred = model_outputs.quantile_outputs
        x_target = model_outputs.normed_target
        loss_mask = model_outputs.loss_mask

        if return_loss:
            x_target = x_target.unsqueeze(-1)
            quantiles = torch.tensor(
                self.backbone.quantile_levels, device=x_target.device
            ).view(1, 1, -1)
            loss = 2 * torch.abs(
                (x_target - q_pred) * ((x_target <= q_pred).float() - quantiles)
            )
            loss = loss * loss_mask.unsqueeze(-1)
            loss = loss.sum(dim=1) / torch.clamp(
                loss_mask.sum(dim=1, keepdim=True), min=1
            )
            loss = loss.sum(dim=-1).mean() / math.sqrt(self.config.num_quantile)
        else:
            loss = None

        x_pred = q_pred.permute(0, 2, 1)
        x_pred = self.backbone.norm_fn.inverse_transform(x_pred)

        return PatchTSTFMPretrainingOutput(
            quantile_outputs=x_pred,
            loss=loss,
            hidden_states=model_outputs.hidden_states,
        )


class PatchTSTFMForPrediction(PatchTSTFMPreTrainedModel):
    main_input_name = "past_values"

    def __init__(self, config: PatchTSTFMConfig):
        super().__init__(config)

        self.config = config
        self.backbone = PatchTSTFMModel(config)

        self._precision = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )

    def model_summary(self) -> str:
        return self.backbone.model_summary()

    def forward(
        self,
        past_values: List[torch.Tensor] | torch.Tensor,
        past_observed_mask: Optional[List[torch.Tensor] | torch.Tensor] = None,
        # future_values: Optional[torch.Tensor] = None,  # future use
        # future_observed_mask: Optional[torch.Tensor] = None,  # future use
        prediction_length: Optional[int] = None,
        quantile_levels: Optional[List[float]] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> PatchTSTFMPredictionOutput:
        forecast_len = (
            prediction_length if prediction_length else self.config.prediction_length
        )
        list_input = isinstance(past_values, list)

        cl = self.config.context_length
        ul = -1
        logger.info(
            f"Context Len: {cl} | Forecast Len: {forecast_len} | Input is tensor: {not list_input}",
        )

        if past_observed_mask is None:
            if list_input:
                past_observed_mask = [~sample.isnan() for sample in past_values]
            else:
                past_observed_mask = ~past_values.isnan()

        fl = max(
            forecast_len,
            ul,
            self.config.d_patch * max(self.config.pretrain_mask_cont, 2),
        )

        if list_input:
            cl = [cl] * len(past_values)
            fl = [fl] * len(past_values)
            forecast_samples, hidden_states = self.forecast_single_step(
                past_values,
                forecast_length=fl,
                observed_inputs_mask=past_observed_mask,
                context_length=cl,
                output_hidden_states=output_hidden_states,
            )
            forecast_samples = [sample[:, :forecast_len] for sample in forecast_samples]
        else:
            if not (
                isinstance(past_values, torch.Tensor)
                and isinstance(past_observed_mask, torch.Tensor)
            ):
                raise ValueError(
                    "Both the `past_values` and `past_observed_mask` should be of type torch.Tensor."
                )

            forecast_samples, hidden_states = self.forecast_single_step_fast(
                past_values,
                forecast_length=fl,
                observed_inputs_mask=past_observed_mask,
                context_length=cl,
                output_hidden_states=output_hidden_states,
            )
            forecast_samples = forecast_samples[:, :, :forecast_len]

        # use internal quantile_levels to compute estimate of mean
        quant_prob = 0.5 - (0.5 - torch.tensor(self.config.quantile_levels)).abs()
        quant_prob /= quant_prob.sum()  # normalize quantile weights

        if not list_input:
            quant_prob = quant_prob.view(1, -1, 1, 1).to(self.device)
            point_forecast: torch.Tensor = (forecast_samples * quant_prob).sum(dim=1)
        else:
            quant_prob = quant_prob.view(-1, 1, 1).to(self.device)
            point_forecast: List[torch.Tensor] = [
                (sample * quant_prob).sum(dim=0) for sample in forecast_samples
            ]

        if quantile_levels is not None:
            try:
                quantile_indices = [
                    self.config.quantile_levels.index(q) for q in quantile_levels
                ]
            except ValueError as e:
                raise ValueError(
                    f"Quantile levels {quantile_levels} not found in model config. Available quantile levels: {self.config.quantile_levels}."
                ) from e
            if list_input:
                forecast_samples = [
                    sample[quantile_indices, :] for sample in forecast_samples
                ]
            else:
                forecast_samples = forecast_samples[:, quantile_indices, :]

        return PatchTSTFMPredictionOutput(
            prediction_outputs=point_forecast,
            quantile_outputs=forecast_samples,
            hidden_states=hidden_states,
        )

    def forecast_single_step_fast(
        self,
        x: torch.Tensor,
        observed_inputs_mask: torch.Tensor,
        forecast_length: int,
        context_length: int,
        output_hidden_states: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Any]:
        # x: batch size x context x features
        # observed_inputs_mask: batch size x context x features
        # forecast_len: list of forecast lengths
        # context_len: list of context lengths
        # output_hidden_states: whether to return hidden states

        miss_mask = ~observed_inputs_mask
        device = x.device

        # x and observed_inputs_mask should be 2d or 3d

        x = x.unsqueeze(-1) if x.ndim == 2 else x
        miss_mask = miss_mask.unsqueeze(-1) if miss_mask.ndim == 2 else miss_mask
        x_mean = x.nanmean(dim=1)  # mean across context dimension

        context_provided = x.shape[1]

        context = min(context_provided + forecast_length, context_length)
        s = context - forecast_length  # part of the context that was provided
        x_in = x[:, -s:, ...]
        miss_mask = miss_mask[:, -s:, ...]
        pad_mask = torch.zeros_like(x_in)

        nan_mask = torch.isnan(x_in)
        x_in = torch.where(nan_mask, x_mean.unsqueeze(1).expand_as(x_in), x_in)

        batch_size, _, n_dim = x_in.shape
        forecast_shape = (batch_size, forecast_length, n_dim)

        pred_mask = torch.cat(
            [torch.zeros_like(x_in), torch.ones(forecast_shape, device=device)], dim=1
        )
        miss_mask = torch.cat(
            [miss_mask, torch.zeros(forecast_shape, device=device)], dim=1
        )
        pad_mask = torch.cat(
            [pad_mask, torch.zeros(forecast_shape, device=device)], dim=1
        )
        x_in = torch.cat(
            [x_in, x_mean.unsqueeze(1).repeat((1, forecast_length, 1))], dim=1
        )

        sample_len = s + forecast_length  # x_in.shape[1]

        if sample_len == self.config.context_length:
            # just pass

            inputs = x_in
            # inputs.append(x_in)
            # pred_mask.append(pred_mask_i)
            # pad_mask.append(pad_mask_i)
            # miss_mask.append(miss_mask_i)
            # time_index.append(time_index_i)
            ts_ends = (0, sample_len)

        elif sample_len < self.config.context_length:
            left_pad = self.config.context_length - sample_len

            pad = x_mean.unsqueeze(dim=1).repeat((1, left_pad, 1))
            inputs = torch.cat(
                (pad, x_in), dim=1
            )  # append left_pad + sample_len = context_len, num_channels

            pred_mask = F.pad(
                pred_mask, (0, 0, left_pad, 0), mode="constant", value=0.0
            )
            pad_mask = F.pad(pad_mask, (0, 0, left_pad, 0), mode="constant", value=1.0)
            miss_mask = F.pad(
                miss_mask, (0, 0, left_pad, 0), mode="constant", value=0.0
            )
            # time_index = F.pad(time_index, (left_pad, 0), mode="constant", value=-1)

            ts_ends = (left_pad, left_pad + sample_len)
            # pad
        else:  # sample_len > self.config.context_length
            # not supported for now
            raise ValueError(
                "Please ensure that provided sample plus the desired forecast is less than the model maximum context length."
            )

        # we are B T N, but backbone wants (B N) T
        inputs = inputs.permute(0, 2, 1).reshape(
            batch_size * inputs.shape[2], inputs.shape[1]
        )
        pred_mask = pred_mask.permute(0, 2, 1).reshape(
            batch_size * pred_mask.shape[2], pred_mask.shape[1]
        )
        miss_mask = miss_mask.permute(0, 2, 1).reshape(
            batch_size * miss_mask.shape[2], miss_mask.shape[1]
        )
        pad_mask = pad_mask.permute(0, 2, 1).reshape(
            batch_size * pad_mask.shape[2], pad_mask.shape[1]
        )

        with torch.autocast(
            device_type=self._device, dtype=self._precision, enabled=True
        ):
            model_output = self.backbone(
                inputs=inputs,
                pred_mask=pred_mask,
                miss_mask=miss_mask,
                pad_mask=pad_mask,
                return_loss=False,
                output_hidden_states=output_hidden_states,
            )
            outputs = model_output.quantile_outputs

        outputs = outputs.permute(0, 2, 1)
        outputs = self.backbone.norm_fn.inverse_transform(outputs)
        outputs = outputs.reshape(batch_size, -1, outputs.shape[1], outputs.shape[2])
        outputs = outputs.permute(0, 2, 3, 1)

        # sample_length <= self.config.context_length
        # ts_ends should always be self.config.context_length

        x_preds = outputs[:, :, ts_ends[0] : ts_ends[1]]
        x_preds = x_preds[:, :, -forecast_length:]

        return x_preds, model_output.hidden_states

    def forecast_single_step(
        self,
        x: List[torch.Tensor] | torch.Tensor,
        observed_inputs_mask: List[torch.Tensor] | torch.Tensor,
        forecast_length: List[int],
        context_length: List[int],
        output_hidden_states: Optional[bool] = False,
    ) -> tuple[list[torch.Tensor], Any]:
        """
        x: list of torch.Tensor of time series, can be of different lengths
        """

        inputs = []
        pad_mask = []
        pred_mask = []
        miss_mask = []
        ts_ends = []
        time_index = []
        sample_lengths = []

        device = x[0].device
        batch_size = len(x)

        # x: batch x time x num_channels
        # x_i: time x num_channels
        # context is full window of input to backbone
        # old_context + forecast = context
        for x_i, observed_inputs_mask_i, c_i, f_i in zip(
            x, observed_inputs_mask, context_length, forecast_length
        ):
            c_i = min(x_i.shape[0] + f_i, c_i)
            s_i = c_i - f_i  # part of the context that was provided
            x_in = x_i[-s_i:]
            x_in = x_in.unsqueeze(-1) if x_in.ndim == 1 else x_in
            miss_mask_i = ~observed_inputs_mask_i[-s_i:]
            miss_mask_i = (
                miss_mask_i.unsqueeze(-1) if miss_mask_i.ndim == 1 else miss_mask_i
            )
            pad_mask_i = torch.zeros_like(x_in)
            x_in_mean = x_in.nanmean(dim=0)

            # Fill NaNs in x_in with corresponding values from x_in_mean for each dimension
            nan_mask = torch.isnan(x_in)
            x_in = torch.where(nan_mask, x_in_mean.unsqueeze(0).expand_as(x_in), x_in)

            f_i_shape = (f_i,) + x_in.shape[1:]

            pred_mask_i = torch.cat(
                [torch.zeros_like(x_in), torch.ones(f_i_shape, device=device)], dim=0
            )
            miss_mask_i = torch.cat(
                [miss_mask_i, torch.zeros(f_i_shape, device=device)], dim=0
            )
            pad_mask_i = torch.cat(
                [pad_mask_i, torch.zeros(f_i_shape, device=device)], dim=0
            )
            x_in = torch.cat(
                [x_in, torch.ones(f_i_shape, device=device) * x_in_mean], dim=0
            )
            sample_len = x_in.shape[0]
            time_index_i = (
                torch.arange(
                    self.config.context_length - sample_len + 1,
                    self.config.context_length + 1,
                ).float()
                / self.config.context_length
            )
            if sample_len == self.config.context_length:
                inputs.append(x_in)
                pred_mask.append(pred_mask_i)
                pad_mask.append(pad_mask_i)
                miss_mask.append(miss_mask_i)
                time_index.append(time_index_i)
                ts_ends.append(torch.tensor([0, sample_len], dtype=torch.int))
                sample_lengths.append(sample_len)
            elif sample_len < self.config.context_length:  # padding
                left_pad = self.config.context_length - sample_len

                # manual pad, since torch pad does not support tensor pad values
                pad = x_in_mean.unsqueeze(dim=0).repeat((left_pad, 1))
                inputs.append(
                    torch.cat((pad, x_in))
                )  # append left_pad + sample_len = context_len, num_channels

                # inputs.append(
                #     F.pad(
                #         x_in,
                #         (left_pad, 0),
                #         mode="constant",
                #         value=x_in.nanmean(dim=0).item(),
                #     )
                # )
                pred_mask.append(
                    F.pad(pred_mask_i, (0, 0, left_pad, 0), mode="constant", value=0.0)
                )
                pad_mask.append(
                    F.pad(pad_mask_i, (0, 0, left_pad, 0), mode="constant", value=1.0)
                )
                miss_mask.append(
                    F.pad(miss_mask_i, (0, 0, left_pad, 0), mode="constant", value=0.0)
                )
                time_index.append(
                    F.pad(time_index_i, (left_pad, 0), mode="constant", value=-1)
                )
                ts_ends.append(
                    torch.tensor([left_pad, left_pad + sample_len], dtype=torch.int)
                )
                sample_lengths.append(sample_len)
            else:  # subsample
                inputs.append(
                    F.interpolate(
                        x_in.view(1, 1, -1),
                        size=self.config.context_length,
                        mode="nearest",
                    ).squeeze()
                )
                pred_mask.append(
                    F.interpolate(
                        pred_mask_i.view(1, 1, -1),
                        size=self.config.context_length,
                        mode="nearest",
                    ).squeeze()
                )
                pad_mask.append(
                    F.interpolate(
                        pad_mask_i.view(1, 1, -1),
                        size=self.config.context_length,
                        mode="nearest",
                    ).squeeze()
                )
                miss_mask.append(
                    F.interpolate(
                        miss_mask_i.view(1, 1, -1),
                        size=self.config.context_length,
                        mode="nearest",
                    ).squeeze()
                )
                time_index.append(
                    F.interpolate(
                        time_index_i.view(1, 1, -1),
                        size=self.config.context_length,
                        mode="nearest",
                    ).squeeze()
                )
                ts_ends.append(
                    torch.tensor([0, self.config.context_length], dtype=torch.int)
                )
                sample_lengths.append(sample_len)

        inputs = torch.stack(inputs, dim=0)
        pred_mask = torch.stack(pred_mask, dim=0)
        pad_mask = torch.stack(pad_mask, dim=0)
        miss_mask = torch.stack(miss_mask, dim=0)
        time_index = torch.stack(time_index, dim=0)
        ts_ends = torch.stack(ts_ends, dim=0)

        # we are B T N, but backbone wants (B N) T
        inputs = inputs.permute(0, 2, 1).reshape(
            batch_size * inputs.shape[2], inputs.shape[1]
        )
        pred_mask = pred_mask.permute(0, 2, 1).reshape(
            batch_size * pred_mask.shape[2], pred_mask.shape[1]
        )
        miss_mask = miss_mask.permute(0, 2, 1).reshape(
            batch_size * miss_mask.shape[2], miss_mask.shape[1]
        )
        pad_mask = pad_mask.permute(0, 2, 1).reshape(
            batch_size * pad_mask.shape[2], pad_mask.shape[1]
        )

        with torch.autocast(
            device_type=self._device, dtype=self._precision, enabled=True
        ):
            model_output = self.backbone(
                inputs=inputs,
                pred_mask=pred_mask,
                miss_mask=miss_mask,
                pad_mask=pad_mask,
                return_loss=False,
                output_hidden_states=output_hidden_states,
            )
            outputs = model_output.quantile_outputs

        outputs = outputs.permute(0, 2, 1)
        outputs = self.backbone.norm_fn.inverse_transform(outputs)
        outputs = outputs.reshape(batch_size, -1, outputs.shape[1], outputs.shape[2])
        outputs = outputs.permute(0, 2, 3, 1)

        x_preds = []
        for i in range(outputs.shape[0]):
            if sample_lengths[i] <= self.config.context_length:
                x_pred = outputs[i][:, ts_ends[i][0] : ts_ends[i][1]]
            else:
                # to do: check me
                x_pred = F.interpolate(
                    outputs[i].unsqueeze(1), size=sample_lengths[i], mode="linear"
                ).squeeze(1)
            x_preds.append(x_pred[:, -forecast_length[i] :])
        return x_preds, model_output.hidden_states
