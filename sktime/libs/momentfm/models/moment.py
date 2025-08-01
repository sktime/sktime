"""Moment model and heads file."""

import logging
import warnings
from argparse import Namespace
from copy import deepcopy
from math import ceil

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.libs.momentfm.common import TASKS
from sktime.libs.momentfm.data.base import TimeseriesOutputs
from sktime.libs.momentfm.models.layers.embed import PatchEmbedding, Patching
from sktime.libs.momentfm.models.layers.revin import RevIN
from sktime.libs.momentfm.utils.masking import Masking
from sktime.libs.momentfm.utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)

SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]

if _check_soft_dependencies(
    ["torch", "huggingface-hub", "transformers"], severity="none"
):
    import torch
    from huggingface_hub import PyTorchModelHubMixin
    from torch import nn
    from transformers import T5Config, T5EncoderModel, T5Model

    class PretrainHead(nn.Module):
        """Pretrained Head."""

        def __init__(
            self,
            d_model: int = 768,
            patch_len: int = 8,
            head_dropout: float = 0.1,
            orth_gain: float = 1.41,
        ):
            super().__init__()
            self.dropout = nn.Dropout(head_dropout)
            self.linear = nn.Linear(d_model, patch_len)

            if orth_gain is not None:
                torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
                self.linear.bias.data.zero_()

        def forward(self, x):
            """Forward Function."""
            x = self.linear(self.dropout(x))
            x = x.flatten(start_dim=2, end_dim=3)
            return x

    class ClassificationHead(nn.Module):
        """Classification Head."""

        def __init__(
            self,
            n_channels: int = 1,
            d_model: int = 768,
            n_classes: int = 2,
            head_dropout: int = 0.1,
            reduction: str = "concat",
        ):
            super().__init__()
            self.dropout = nn.Dropout(head_dropout)
            if reduction == "mean":
                self.linear = nn.Linear(d_model, n_classes)
            elif reduction == "concat":
                self.linear = nn.Linear(n_channels * d_model, n_classes)
            else:
                raise ValueError(
                    f"Reduction method {reduction} not implemented. "
                    f"Only 'mean' and 'concat' are supported."
                )

        def forward(self, x, input_mask: torch.Tensor = None):
            """Forward Function."""
            x = torch.mean(x, dim=1)
            x = self.dropout(x)
            y = self.linear(x)
            return y

    class ForecastingHead(nn.Module):
        """Forecasting Head."""

        def __init__(
            self,
            head_nf: int = 768 * 64,
            forecast_horizon: int = 96,
            head_dropout: int = 0,
        ):
            super().__init__()
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(head_dropout)
            self.linear = nn.Linear(head_nf, forecast_horizon)

        def forward(self, x, input_mask: torch.Tensor = None):
            """Forward Function."""
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
            return x

    class MOMENT(nn.Module):
        """Moment Class."""

        def __init__(self, config, **kwargs: dict):
            super().__init__()
            config = self._update_inputs(config, **kwargs)
            config = self._validate_inputs(config)
            self.config = config
            self.task_name = config.task_name
            self.seq_len = config.seq_len
            self.patch_len = config.patch_len

            self.normalizer = RevIN(
                num_features=1, affine=config.getattr("revin_affine", False)
            )
            self.tokenizer = Patching(
                patch_len=config.patch_len, stride=config.patch_stride_len
            )
            self.patch_embedding = PatchEmbedding(
                d_model=config.d_model,
                seq_len=config.seq_len,
                patch_len=config.patch_len,
                stride=config.patch_stride_len,
                dropout=config.getattr("dropout", 0.1),
                add_positional_embedding=config.getattr(
                    "add_positional_embedding", True
                ),
                value_embedding_bias=config.getattr("value_embedding_bias", False),
                orth_gain=config.getattr("orth_gain", 1.41),
            )
            self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
            self.encoder = self._get_transformer_backbone(config)
            self.head = self._get_head(self.task_name)

            # Frozen parameters
            self.freeze_embedder = config.getattr("freeze_embedder", True)
            self.freeze_encoder = config.getattr("freeze_encoder", True)
            self.freeze_head = config.getattr("freeze_head", False)

            if self.freeze_embedder:
                self.patch_embedding = freeze_parameters(self.patch_embedding)
            if self.freeze_encoder:
                self.encoder = freeze_parameters(self.encoder)
            if self.freeze_head:
                self.head = freeze_parameters(self.head)

        def _update_inputs(self, config, **kwargs: dict) -> NamespaceWithDefaults:
            """Update Inputs."""
            if isinstance(config, dict) and "model_kwargs" in kwargs:
                return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
            else:
                return NamespaceWithDefaults.from_namespace(config)

        def _validate_inputs(
            self, config: NamespaceWithDefaults
        ) -> NamespaceWithDefaults:
            """Validate Inputs."""
            if (
                config.d_model is None
                and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
            ):
                config.d_model = get_huggingface_model_dimensions(
                    config.transformer_backbone
                )
                logging.info(f"Setting d_model to {config.d_model}")
            elif config.d_model is None:
                raise ValueError(
                    "d_model must be specified if transformer backbone "
                    "unless transformer backbone is a Huggingface model."
                )

            if config.transformer_type not in [
                "encoder_only",
                "decoder_only",
                "encoder_decoder",
            ]:
                raise ValueError(
                    "transformer_type must be one of "
                    "['encoder_only', 'decoder_only', 'encoder_decoder']"
                )

            if config.patch_stride_len != config.patch_len:
                warnings.warn("Patch stride length is not equal to patch length.")
            return config

        def _get_head(self, task_name: str) -> nn.Module:
            if task_name == TASKS.RECONSTRUCTION:
                return PretrainHead(
                    self.config.d_model,
                    self.config.patch_len,
                    self.config.getattr("dropout", 0.1),
                    self.config.getattr("orth_gain", 1.41),
                )
            elif task_name == TASKS.CLASSIFICATION:
                return ClassificationHead(
                    self.config.n_channels,
                    self.config.d_model,
                    self.config.num_class,
                    self.config.getattr("dropout", 0.1),
                    reduction=self.config.getattr("reduction", "concat"),
                )
            elif task_name == TASKS.FORECASTING:
                num_patches = (
                    max(self.config.seq_len, self.config.patch_len)
                    - self.config.patch_len
                ) // self.config.patch_stride_len + 1
                self.head_nf = self.config.d_model * num_patches
                return ForecastingHead(
                    self.head_nf,
                    self.config.forecast_horizon,
                    self.config.getattr("head_dropout", 0.1),
                )
            elif task_name == TASKS.EMBED:
                return nn.Identity()
            else:
                raise NotImplementedError(f"Task {task_name} not implemented.")

        def _get_transformer_backbone(self, config) -> nn.Module:
            if config.getattr("randomly_initialize_backbone", False):
                model_config = T5Config.from_pretrained(config.transformer_backbone)
                transformer_backbone = T5Model(model_config)
                logging.info(
                    f"Initializing randomly initialized transformer from "
                    f"{config.transformer_backbone}."
                )
            else:
                transformer_backbone = T5EncoderModel.from_pretrained(
                    config.transformer_backbone
                )
                logging.info(
                    f"Initializing pre-trained transformer from "
                    f"{config.transformer_backbone}."
                )

            transformer_backbone = transformer_backbone.get_encoder()

            if config.getattr("enable_gradient_checkpointing", True):
                transformer_backbone.gradient_checkpointing_enable()
                logging.info("Enabling gradient checkpointing.")

            return transformer_backbone

        def __call__(self, *args, **kwargs) -> TimeseriesOutputs:
            """__call__ function."""
            return self.forward(*args, **kwargs)

        def embed(
            self,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor = None,
            reduction: str = "mean",
            **kwargs,
        ) -> TimeseriesOutputs:
            """Embed function."""
            batch_size, n_channels, seq_len = x_enc.shape

            if input_mask is None:
                input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

            x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
            x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

            input_mask_patch_view = Masking.convert_seq_to_patch_view(
                input_mask, self.patch_len
            )

            x_enc = self.tokenizer(x=x_enc)
            enc_in = self.patch_embedding(x_enc, mask=input_mask)

            n_patches = enc_in.shape[2]
            enc_in = enc_in.reshape(
                (batch_size * n_channels, n_patches, self.config.d_model)
            )

            patch_view_mask = Masking.convert_seq_to_patch_view(
                input_mask, self.patch_len
            )
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
            enc_out = outputs.last_hidden_state

            enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
            # [batch_size x n_channels x n_patches x d_model]

            if reduction == "mean":
                enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
                # [batch_size x n_patches x d_model]
                input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                    1, 1, self.config.d_model
                )
                enc_out = (input_mask_patch_view * enc_out).sum(
                    dim=1
                ) / input_mask_patch_view.sum(dim=1)
            else:
                raise NotImplementedError(
                    f"Reduction method {reduction} not implemented."
                )

            return TimeseriesOutputs(
                embeddings=enc_out, input_mask=input_mask, metadata=reduction
            )

        def reconstruction(
            self,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor = None,
            mask: torch.Tensor = None,
            **kwargs,
        ) -> TimeseriesOutputs:
            """Reconstruction Function."""
            batch_size, n_channels, _ = x_enc.shape

            if mask is None:
                mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
                mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

            x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
            # Prevent too short time-series from causing NaNs
            x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

            x_enc = self.tokenizer(x=x_enc)
            enc_in = self.patch_embedding(x_enc, mask=mask)

            n_patches = enc_in.shape[2]
            enc_in = enc_in.reshape(
                (batch_size * n_channels, n_patches, self.config.d_model)
            )

            patch_view_mask = Masking.convert_seq_to_patch_view(
                input_mask, self.patch_len
            )
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
            if self.config.transformer_type == "encoder_decoder":
                outputs = self.encoder(
                    inputs_embeds=enc_in,
                    decoder_inputs_embeds=enc_in,
                    attention_mask=attention_mask,
                )
            else:
                outputs = self.encoder(
                    inputs_embeds=enc_in, attention_mask=attention_mask
                )
            enc_out = outputs.last_hidden_state

            enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

            dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
            dec_out = self.normalizer(x=dec_out, mode="denorm")

            if self.config.getattr("debug", False):
                illegal_output = self._check_model_weights_for_illegal_values()
            else:
                illegal_output = None

            return TimeseriesOutputs(
                input_mask=input_mask,
                reconstruction=dec_out,
                pretrain_mask=mask,
                illegal_output=illegal_output,
            )

        def reconstruct(
            self,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor = None,
            mask: torch.Tensor = None,
            **kwargs,
        ) -> TimeseriesOutputs:
            """Reconstruct Function."""
            if mask is None:
                mask = torch.ones_like(input_mask)

            batch_size, n_channels, _ = x_enc.shape
            x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")

            x_enc = self.tokenizer(x=x_enc)
            enc_in = self.patch_embedding(x_enc, mask=mask)

            n_patches = enc_in.shape[2]
            enc_in = enc_in.reshape(
                (batch_size * n_channels, n_patches, self.config.d_model)
            )
            # [batch_size * n_channels x n_patches x d_model]

            patch_view_mask = Masking.convert_seq_to_patch_view(
                input_mask, self.patch_len
            )
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0).to(
                x_enc.device
            )

            n_tokens = 0
            if "prompt_embeds" in kwargs:
                prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

                if isinstance(prompt_embeds, nn.Embedding):
                    prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

                n_tokens = prompt_embeds.shape[1]

                enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)
                attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

            if self.config.transformer_type == "encoder_decoder":
                outputs = self.encoder(
                    inputs_embeds=enc_in,
                    decoder_inputs_embeds=enc_in,
                    attention_mask=attention_mask,
                )
            else:
                outputs = self.encoder(
                    inputs_embeds=enc_in, attention_mask=attention_mask
                )
            enc_out = outputs.last_hidden_state
            enc_out = enc_out[:, n_tokens:, :]

            enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
            # [batch_size x n_channels x n_patches x d_model]

            dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
            dec_out = self.normalizer(x=dec_out, mode="denorm")

            return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

        def detect_anomalies(
            self,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor = None,
            anomaly_criterion: str = "mse",
            **kwargs,
        ) -> TimeseriesOutputs:
            """Detect Anomalies Function."""
            outputs = self.reconstruct(x_enc=x_enc, input_mask=input_mask)
            self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

            anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

            return TimeseriesOutputs(
                input_mask=input_mask,
                reconstruction=outputs.reconstruction,
                anomaly_scores=anomaly_scores,
                metadata={"anomaly_criterion": anomaly_criterion},
            )

        def forecast(
            self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, **kwargs
        ) -> TimeseriesOutputs:
            """Forecast Function."""
            batch_size, n_channels, seq_len = x_enc.shape

            x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
            x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

            x_enc = self.tokenizer(x=x_enc)
            enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))

            n_patches = enc_in.shape[2]
            enc_in = enc_in.reshape(
                (batch_size * n_channels, n_patches, self.config.d_model)
            )

            patch_view_mask = Masking.convert_seq_to_patch_view(
                input_mask, self.patch_len
            )
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
            enc_out = outputs.last_hidden_state
            enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
            # [batch_size x n_channels x n_patches x d_model]

            dec_out = self.head(enc_out)  # [batch_size x n_channels x forecast_horizon]
            dec_out = self.normalizer(x=dec_out, mode="denorm")

            return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)

        def short_forecast(
            self,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor = None,
            forecast_horizon: int = 1,
            **kwargs,
        ) -> TimeseriesOutputs:
            """Short_forecast Function."""
            batch_size, n_channels, seq_len = x_enc.shape
            num_masked_patches = ceil(forecast_horizon / self.patch_len)
            num_masked_timesteps = num_masked_patches * self.patch_len

            x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
            x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

            # Shift the time-series and mask the last few timesteps for forecasting
            x_enc = torch.roll(x_enc, shifts=-num_masked_timesteps, dims=2)
            input_mask = torch.roll(input_mask, shifts=-num_masked_timesteps, dims=1)

            # Attending to mask tokens
            input_mask[:, -num_masked_timesteps:] = 1
            mask = torch.ones_like(input_mask)
            mask[:, -num_masked_timesteps:] = 0

            x_enc = self.tokenizer(x=x_enc)
            enc_in = self.patch_embedding(x_enc, mask=mask)

            n_patches = enc_in.shape[2]
            enc_in = enc_in.reshape(
                (batch_size * n_channels, n_patches, self.config.d_model)
            )
            # [batch_size * n_channels x n_patches x d_model]

            patch_view_mask = Masking.convert_seq_to_patch_view(
                input_mask, self.patch_len
            )
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
            enc_out = outputs.last_hidden_state
            enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

            dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]

            end = -num_masked_timesteps + forecast_horizon
            end = None if end == 0 else end

            dec_out = self.normalizer(x=dec_out, mode="denorm")
            forecast = dec_out[:, :, -num_masked_timesteps:end]

            return TimeseriesOutputs(
                input_mask=input_mask,
                reconstruction=dec_out,
                forecast=forecast,
                metadata={"forecast_horizon": forecast_horizon},
            )

        def classify(
            self,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor = None,
            reduction: str = "concat",
            **kwargs,
        ) -> TimeseriesOutputs:
            """Classify Function."""
            batch_size, n_channels, seq_len = x_enc.shape

            if input_mask is None:
                input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

            x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
            x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

            # input_mask_patch_view = Masking.convert_seq_to_patch_view(
            #     input_mask, self.patch_len
            # )

            x_enc = self.tokenizer(x=x_enc)
            enc_in = self.patch_embedding(x_enc, mask=input_mask)

            n_patches = enc_in.shape[2]
            enc_in = enc_in.reshape(
                (batch_size * n_channels, n_patches, self.config.d_model)
            )

            patch_view_mask = Masking.convert_seq_to_patch_view(
                input_mask, self.patch_len
            )
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
            enc_out = outputs.last_hidden_state

            enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
            # [batch_size x n_channels x n_patches x d_model]

            # Mean across channels
            if reduction == "mean":
                # [batch_size x n_patches x d_model]
                enc_out = enc_out.mean(dim=1, keepdim=False)
            # Concatenate across channels
            elif reduction == "concat":
                # [batch_size x n_patches x d_model * n_channels]
                enc_out = enc_out.permute(0, 2, 3, 1).reshape(
                    batch_size, n_patches, self.config.d_model * n_channels
                )

            else:
                raise NotImplementedError(
                    f"Reduction method {reduction} not implemented."
                )

            logits = self.head(enc_out, input_mask=input_mask)

            return TimeseriesOutputs(
                embeddings=enc_out, logits=logits, metadata=reduction
            )

        def forward(
            self,
            x_enc: torch.Tensor,
            mask: torch.Tensor = None,
            input_mask: torch.Tensor = None,
            **kwargs,
        ) -> TimeseriesOutputs:
            """Forward Function."""
            if input_mask is None:
                input_mask = torch.ones_like(x_enc[:, 0, :])

            if self.task_name == TASKS.RECONSTRUCTION:
                return self.reconstruction(
                    x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs
                )
            elif self.task_name == TASKS.EMBED:
                return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.FORECASTING:
                return self.forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.CLASSIFICATION:
                return self.classify(x_enc=x_enc, input_mask=input_mask, **kwargs)
            else:
                raise NotImplementedError(f"Task {self.task_name} not implemented.")

    class MOMENTPipeline(MOMENT, PyTorchModelHubMixin):
        """Moment Pipeline."""

        def __init__(self, config, **kwargs: dict):
            self._validate_model_kwargs(**kwargs)
            self.new_task_name = kwargs.get("model_kwargs", {}).pop(
                "task_name", TASKS.RECONSTRUCTION
            )
            super().__init__(config, **kwargs)

        def _validate_model_kwargs(self, **kwargs: dict) -> None:
            kwargs = deepcopy(kwargs)
            kwargs.setdefault("model_kwargs", {"task_name": TASKS.RECONSTRUCTION})
            kwargs["model_kwargs"].setdefault("task_name", TASKS.RECONSTRUCTION)
            config = Namespace(**kwargs["model_kwargs"])

            if config.task_name == TASKS.FORECASTING:
                if not hasattr(config, "forecast_horizon"):
                    raise ValueError(
                        "forecast_horizon must be specified for long-horizon "
                        "forecasting."
                    )

            if config.task_name == TASKS.CLASSIFICATION:
                if not hasattr(config, "n_channels"):
                    raise ValueError("n_channels must be specified for classification.")
                if not hasattr(config, "num_class"):
                    raise ValueError("num_class must be specified for classification.")

        def init(self) -> None:
            """Init function."""
            if self.new_task_name != TASKS.RECONSTRUCTION:
                self.task_name = self.new_task_name
                self.head = self._get_head(self.new_task_name)

    def freeze_parameters(model):
        """Freeze parameters of the model."""
        # Freeze the parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        return model
else:

    class PretrainHead:
        """Dummy class if soft dependencies are not available."""

        pass

    class ClassificationHead:
        """Dummy class if soft dependencies are not available."""

        pass

    class ForecastingHead:
        """Dummy class if soft dependencies are not available."""

        pass

    class MOMENT:
        """Dummy class if soft dependencies are not available."""

        pass

    class MOMENTPipeline:
        """Dummy class if soft dependencies are not available."""

        pass
