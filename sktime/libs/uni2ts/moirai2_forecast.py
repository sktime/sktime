#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.utils.dependencies import _safe_import

if _check_soft_dependencies("lightning", severity="none"):
    import lightning as L

else:

    class L:
        class LightningModule:
            pass


if _check_soft_dependencies("torch", severity="none"):
    import torch

    from .moirai2_module import Moirai2Module

if _check_soft_dependencies("einops", severity="none"):
    from einops import rearrange, reduce, repeat

Input = _safe_import("gluonts.model.Input")
InputSpec = _safe_import("gluonts.model.InputSpec")

PyTorchPredictor = _safe_import("gluonts.torch.PyTorchPredictor")
QuantileForecastGenerator = _safe_import(
    "gluonts.model.forecast_generator.QuantileForecastGenerator"
)

AddObservedValuesIndicator = _safe_import(
    "gluonts.transform.AddObservedValuesIndicator"
)
AsNumpyArray = _safe_import("gluonts.transform.AsNumpyArray")
CausalMeanValueImputation = _safe_import(
    "gluonts.transform.CausalMeanValueImputation"
)
ExpandDimArray = _safe_import("gluonts.transform.ExpandDimArray")
TestSplitSampler = _safe_import("gluonts.transform.TestSplitSampler")

TFTInstanceSplitter = _safe_import("gluonts.transform.split.TFTInstanceSplitter")


class CausalMeanImputation:
    """Replace NaNs with causal running mean, with last-value backfill."""

    value = 0.0

    def __call__(self, x):
        mask = np.isnan(x).T

        # last-value backfill first
        x_t = x.T
        if x_t.ndim == 1:
            x_t = x_t[np.newaxis, :]
        for row in x_t:
            if np.isnan(row[0]):
                row[0] = self.value
            for i in range(1, len(row)):
                if np.isnan(row[i]):
                    row[i] = row[i - 1]
        x = x_t.T if x.ndim > 1 else x_t.squeeze(0)
        mask[0] = False
        x = x.T if x.ndim > 1 else x

        if x.ndim == 1:
            adjusted = np.concatenate((np.array([0.0]), x[:-1]))
            cumsum = np.cumsum(adjusted)
            indices = np.arange(len(x), dtype=float)
            indices[0] = 1.0
            ar_res = cumsum / indices
            x[mask.ravel()] = ar_res[mask.ravel()]
        else:
            adjusted = np.vstack((np.zeros((1, x.shape[1])), x[:-1, :]))
            cumsum = np.cumsum(adjusted, axis=0)
            indices = np.arange(len(x), dtype=float).reshape(-1, 1)
            indices[0] = 1.0
            ar_res = cumsum / indices
            x[mask] = ar_res[mask]
        return x.T if x.ndim > 1 else x


class Moirai2Forecast(L.LightningModule):
    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        context_length: int,
        module_kwargs: dict | None = None,
        module: Moirai2Module | None = None,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        if module_kwargs and "attn_dropout_p" in module_kwargs:
            module_kwargs["attn_dropout_p"] = 0
        if module_kwargs and "dropout_p" in module_kwargs:
            module_kwargs["dropout_p"] = 0

        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = Moirai2Module(**module_kwargs) if module is None else module
        self.module.eval()

    @contextmanager
    def hparams_context(
        self,
        prediction_length=None,
        target_dim=None,
        feat_dynamic_real_dim=None,
        past_feat_dynamic_real_dim=None,
        context_length=None,
    ):
        kwargs = {
            "prediction_length": prediction_length,
            "target_dim": target_dim,
            "feat_dynamic_real_dim": feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "context_length": context_length,
        }
        old_hparams = deepcopy(self.hparams)
        for kw, arg in kwargs.items():
            if arg is not None:
                self.hparams[kw] = arg

        yield self

        for kw in kwargs:
            self.hparams[kw] = old_hparams[kw]

    def create_predictor(self, batch_size: int, device: str = "auto"):
        ts_fields = []
        if self.hparams.feat_dynamic_real_dim > 0:
            ts_fields.append("feat_dynamic_real")
            ts_fields.append("observed_feat_dynamic_real")
        past_ts_fields = []
        if self.hparams.past_feat_dynamic_real_dim > 0:
            past_ts_fields.append("past_feat_dynamic_real")
            past_ts_fields.append("past_observed_feat_dynamic_real")
        instance_splitter = TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),
            past_length=self.past_length,
            future_length=self.hparams.prediction_length,
            observed_value_field="observed_target",
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )
        return PyTorchPredictor(
            input_names=self.prediction_input_names,
            prediction_net=self,
            batch_size=batch_size,
            prediction_length=self.hparams.prediction_length,
            input_transform=self.get_default_transform() + instance_splitter,
            forecast_generator=QuantileForecastGenerator(
                self.module.quantile_levels
            ),
            device=device,
        )

    def describe_inputs(self, batch_size=1):
        data = {
            "past_target": Input(
                shape=(batch_size, self.past_length, self.hparams.target_dim),
                dtype=torch.float,
            ),
            "past_observed_target": Input(
                shape=(batch_size, self.past_length, self.hparams.target_dim),
                dtype=torch.bool,
            ),
            "past_is_pad": Input(
                shape=(batch_size, self.past_length),
                dtype=torch.bool,
            ),
        }
        if self.hparams.feat_dynamic_real_dim > 0:
            data["feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.hparams.prediction_length,
                    self.hparams.feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.hparams.prediction_length,
                    self.hparams.feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        if self.hparams.past_feat_dynamic_real_dim > 0:
            data["past_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.past_feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["past_observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.past_feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        return InputSpec(data=data, zeros_fn=torch.zeros)

    @property
    def prediction_input_names(self):
        return list(self.describe_inputs())

    @property
    def training_input_names(self):
        return self.prediction_input_names + [
            "future_target",
            "future_observed_values",
        ]

    @property
    def past_length(self) -> int:
        return self.hparams.context_length

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil(self.hparams.context_length / patch_size)

    def prediction_token_length(self, patch_size) -> int:
        return math.ceil(self.hparams.prediction_length / patch_size)

    def forward(
        self,
        past_target,
        past_observed_target,
        past_is_pad,
        feat_dynamic_real=None,
        observed_feat_dynamic_real=None,
        past_feat_dynamic_real=None,
        past_observed_feat_dynamic_real=None,
    ):
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            self.module.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        per_var_context_token = self.context_token_length(self.module.patch_size)
        total_context_token = self.hparams.target_dim * per_var_context_token
        per_var_predict_token = self.prediction_token_length(self.module.patch_size)
        total_predict_token = self.hparams.target_dim * per_var_predict_token

        pred_index = torch.arange(
            start=per_var_context_token - 1,
            end=total_context_token,
            step=per_var_context_token,
        )
        assign_index = torch.arange(
            start=total_context_token,
            end=total_context_token + total_predict_token,
            step=per_var_predict_token,
        )
        quantile_prediction = repeat(
            target,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.module.quantile_levels),
            patch_size=self.module.patch_size,
        ).clone()

        preds = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            training_mode=False,
        )

        def structure_multi_predict(
            per_var_predict_token,
            pred_index,
            assign_index,
            preds,
        ):
            preds = rearrange(
                preds,
                "... (predict_token num_quantiles patch_size)"
                " -> ... predict_token num_quantiles patch_size",
                predict_token=self.module.num_predict_token,
                num_quantiles=self.module.num_quantiles,
                patch_size=self.module.patch_size,
            )
            preds = rearrange(
                preds[..., pred_index, :per_var_predict_token, :, :],
                "... pred_index predict_token num_quantiles patch_size"
                " -> ... (pred_index predict_token) num_quantiles patch_size",
            )
            adjusted_assign_index = torch.cat(
                [
                    torch.arange(start=idx, end=idx + per_var_predict_token)
                    for idx in assign_index
                ]
            )
            return preds, adjusted_assign_index

        if per_var_predict_token <= self.module.num_predict_token:
            preds, adjusted_assign_index = structure_multi_predict(
                per_var_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction[..., adjusted_assign_index, :, :] = preds
            preds_out = self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )
            return (preds_out,), None, None
        else:
            expand_target = repeat(
                target,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_prediction_mask = repeat(
                prediction_mask,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_observed_mask = repeat(
                observed_mask,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_sample_id = repeat(
                sample_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_time_id = repeat(
                time_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_variate_id = repeat(
                variate_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()

            preds, adjusted_assign_index = structure_multi_predict(
                self.module.num_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction[..., adjusted_assign_index, :, :] = preds

            expand_target[..., adjusted_assign_index, :] = rearrange(
                preds,
                "... predict_token num_quantiles patch_size"
                " -> ... num_quantiles predict_token patch_size",
                num_quantiles=self.module.num_quantiles,
                patch_size=self.module.patch_size,
                predict_token=self.module.num_predict_token,
            )
            expand_prediction_mask[..., adjusted_assign_index] = False

            remain_step = per_var_predict_token - self.module.num_predict_token
            while remain_step > 0:
                preds = self.module(
                    expand_target,
                    expand_observed_mask,
                    expand_sample_id,
                    expand_time_id,
                    expand_variate_id,
                    expand_prediction_mask,
                    training_mode=False,
                )

                pred_index = assign_index + self.module.num_predict_token - 1
                assign_index = pred_index + 1
                preds, adjusted_assign_index = structure_multi_predict(
                    (
                        self.module.num_predict_token
                        if remain_step - self.module.num_predict_token > 0
                        else remain_step
                    ),
                    pred_index,
                    assign_index,
                    preds,
                )
                quantile_prediction_next_step = rearrange(
                    preds,
                    "... num_quantiles_prev pred_index num_quantiles patch_size"
                    " -> ... pred_index (num_quantiles_prev num_quantiles)"
                    " patch_size",
                    num_quantiles=self.module.num_quantiles,
                    patch_size=self.module.patch_size,
                )
                quantile_prediction_next_step = torch.quantile(
                    quantile_prediction_next_step,
                    torch.tensor(
                        self.module.quantile_levels,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    dim=-2,
                )
                quantile_prediction[..., adjusted_assign_index, :, :] = rearrange(
                    quantile_prediction_next_step,
                    "num_quantiles ... patch_size"
                    " -> ... num_quantiles patch_size",
                )

                expand_target[..., adjusted_assign_index, :] = rearrange(
                    quantile_prediction_next_step,
                    "num_quantiles batch_size predict_token patch_size"
                    " -> batch_size num_quantiles predict_token patch_size",
                    num_quantiles=self.module.num_quantiles,
                    patch_size=self.module.patch_size,
                    predict_token=len(adjusted_assign_index),
                )
                expand_prediction_mask[..., adjusted_assign_index] = False

                remain_step -= self.module.num_predict_token

            preds_out = self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )
            return (preds_out,), None, None

    @staticmethod
    def _patched_seq_pad(patch_size, x, dim, left=True, value=None):
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(self, patch_size, past_observed_target):
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(
            past_seq_id.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0
        )
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    def _convert(
        self,
        patch_size,
        past_target,
        past_observed_target,
        past_is_pad,
        future_target=None,
        future_observed_target=None,
        future_is_pad=None,
        feat_dynamic_real=None,
        observed_feat_dynamic_real=None,
        past_feat_dynamic_real=None,
        past_observed_feat_dynamic_real=None,
    ):
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        if future_target is None:
            future_target = torch.zeros(
                batch_shape + (self.hparams.prediction_length, past_target.shape[-1]),
                dtype=past_target.dtype,
                device=device,
            )
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (self.hparams.prediction_length, past_observed_target.shape[-1]),
                dtype=torch.bool,
                device=device,
            )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.hparams.prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )
        dim_count += past_target.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (self.context_token_length(patch_size) * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        if feat_dynamic_real is not None:
            if observed_feat_dynamic_real is None:
                raise ValueError(
                    "observed_feat_dynamic_real must be provided "
                    "if feat_dynamic_real is provided"
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                ]
            )
            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                ]
            )
            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=feat_dynamic_real.shape[-1],
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            time_id.extend(
                [past_seq_id] * feat_dynamic_real.shape[-1]
                + [future_seq_id] * feat_dynamic_real.shape[-1]
            )
            variate_id.extend(
                [
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=self.context_token_length(patch_size),
                    ),
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=self.prediction_token_length(patch_size),
                    ),
                ]
            )
            dim_count += feat_dynamic_real.shape[-1]
            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            self.context_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

        if past_feat_dynamic_real is not None:
            if past_observed_feat_dynamic_real is None:
                raise ValueError(
                    "past_observed_feat_dynamic_real must be provided "
                    "if past_feat_dynamic_real is provided"
                )
            target.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                )
            )
            observed_mask.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size,
                            past_observed_feat_dynamic_real,
                            -2,
                            left=True,
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                )
            )
            sample_id.append(
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_feat_dynamic_real.shape[-1],
                )
            )
            time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])

            variate_id.append(
                repeat(
                    torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                )
            )
            dim_count += past_feat_dynamic_real.shape[-1]
            prediction_mask.append(
                torch.zeros(
                    batch_shape
                    + (
                        self.context_token_length(patch_size)
                        * past_feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

    def _format_preds(self, num_quantiles, patch_size, preds, target_dim):
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :num_quantiles, :patch_size]
        preds = rearrange(
            preds,
            "... (dim seq) num_quantiles patch -> ... (seq patch) num_quantiles dim",
            dim=target_dim,
        )[..., : self.hparams.prediction_length, :, :]
        return preds.squeeze(-1)

    def get_default_transform(self):
        transform = AsNumpyArray(
            field="target",
            expected_ndim=1 if self.hparams.target_dim == 1 else 2,
            dtype=np.float32,
        )
        if self.hparams.target_dim == 1:
            transform += AddObservedValuesIndicator(
                target_field="target",
                output_field="observed_target",
                imputation_method=CausalMeanValueImputation(),
                dtype=bool,
            )
            transform += ExpandDimArray(field="target", axis=0)
            transform += ExpandDimArray(field="observed_target", axis=0)
        else:
            transform += AddObservedValuesIndicator(
                target_field="target",
                output_field="observed_target",
                dtype=bool,
            )

        if self.hparams.feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="feat_dynamic_real",
                output_field="observed_feat_dynamic_real",
                dtype=bool,
            )

        if self.hparams.past_feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="past_feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="past_feat_dynamic_real",
                output_field="past_observed_feat_dynamic_real",
                dtype=bool,
            )
        return transform
