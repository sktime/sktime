# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("lightning"):
    from lightning import LightningModule

if _check_soft_dependencies("torch"):
    import torch
    import torch.nn.functional as F

if _check_soft_dependencies("gluonts"):
    from gluonts.core.component import validated
    from gluonts.itertools import prod
    from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
    from gluonts.torch.util import repeat_along_dim, take_last


from ..data.augmentations.augmentations import (
    ApplyAugmentations,
    Jitter,
    MagnitudeWarp,
    Permutation,
    Rotation,
    Scaling,
    TimeWarp,
    WindowSlice,
    WindowWarp,
)
from ..data.augmentations.freq_mask import freq_mask
from ..data.augmentations.freq_mix import freq_mix
from ..gluon_utils.gluon_ts_distributions.implicit_quantile_network import (
    ImplicitQuantileNetworkOutput,
)
from ..model.module import LagLlamaModel


class LagLlamaLightningModule(LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``LagLlamaLightningModule`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``LagLlamaLightningModule`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``LagLlamaLightningModule`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        context_length: int,
        prediction_length: int,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        freq_mask_rate: float = 0.1,
        freq_mixing_rate: float = 0.1,
        jitter_prob: float = 0.0,
        jitter_sigma: float = 0.03,
        scaling_prob: float = 0.0,
        scaling_sigma: float = 0.1,
        rotation_prob: float = 0.0,
        permutation_prob: float = 0.0,
        permutation_max_segments: int = 5,
        permutation_seg_mode: str = "equal",
        magnitude_warp_prob: float = 0.0,
        magnitude_warp_sigma: float = 0.2,
        magnitude_warp_knot: int = 4,
        time_warp_prob: float = 0.0,
        time_warp_sigma: float = 0.2,
        time_warp_knot: int = 4,
        window_slice_prob: float = 0.0,
        window_slice_reduce_ratio: float = 0.9,
        window_warp_prob: float = 0.0,
        window_warp_window_ratio: float = 0.1,
        window_warp_scales: list = [0.5, 2.0],
        data_id_to_name_map: dict = {},
        use_cosine_annealing_lr: bool = False,
        cosine_annealing_lr_args: dict = {},
        track_loss_per_series: bool = False,
        nonnegative_pred_samples: bool = False,
        use_kv_cache: bool = True,
        use_single_pass_sampling: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context_length = self.hparams.context_length
        self.prediction_length = self.hparams.prediction_length
        self.model = LagLlamaModel(**self.hparams.model_kwargs)
        self.loss = self.hparams.loss
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.freq_mask_rate = self.hparams.freq_mask_rate
        self.freq_mixing_rate = self.hparams.freq_mixing_rate
        self.jitter_prob = self.hparams.jitter_prob
        self.jitter_sigma = self.hparams.jitter_sigma
        self.scaling_prob = self.hparams.scaling_prob
        self.scaling_sigma = self.hparams.scaling_sigma
        self.rotation_prob = self.hparams.rotation_prob
        self.permutation_prob = self.hparams.permutation_prob
        self.permutation_max_segments = self.hparams.permutation_max_segments
        self.permutation_seg_mode = self.hparams.permutation_seg_mode
        self.magnitude_warp_prob = self.hparams.magnitude_warp_prob
        self.magnitude_warp_sigma = self.hparams.magnitude_warp_sigma
        self.magnitude_warp_knot = self.hparams.magnitude_warp_knot
        self.time_warp_prob = self.hparams.time_warp_prob
        self.time_warp_sigma = self.hparams.time_warp_sigma
        self.time_warp_knot = self.hparams.time_warp_knot
        self.window_slice_prob = self.hparams.window_slice_prob
        self.window_slice_reduce_ratio = self.hparams.window_slice_reduce_ratio
        self.window_warp_prob = self.hparams.window_warp_prob
        self.window_warp_window_ratio = self.hparams.window_warp_window_ratio
        self.window_warp_scales = self.hparams.window_warp_scales
        self.data_id_to_name_map = self.hparams.data_id_to_name_map
        self.use_cosine_annealing_lr = self.hparams.use_cosine_annealing_lr
        self.cosine_annealing_lr_args = self.hparams.cosine_annealing_lr_args
        self.track_loss_per_series = self.hparams.track_loss_per_series
        self.nonnegative_pred_samples = self.hparams.nonnegative_pred_samples

        self.time_feat = self.hparams.model_kwargs["time_feat"]
        # data_id based
        self.train_loss_dict = {}
        self.val_loss_dict = {}
        # item_id based - to be used only in single-dataset mode
        self.train_loss_dict_per_series = {}
        self.val_loss_dict_per_series = {}
        self.use_kv_cache = use_kv_cache
        self.use_single_pass_sampling = use_single_pass_sampling
        self.transforms = []
        aug_probs = dict(
            Jitter=dict(prob=self.jitter_prob, sigma=self.jitter_sigma),
            Scaling=dict(prob=self.scaling_prob, sigma=self.scaling_sigma),
            Rotation=dict(prob=self.rotation_prob),
            Permutation=dict(
                prob=self.permutation_prob,
                max_segments=self.permutation_max_segments,
                seg_mode=self.permutation_seg_mode,
            ),
            MagnitudeWarp=dict(
                prob=self.magnitude_warp_prob,
                sigma=self.magnitude_warp_sigma,
                knot=self.magnitude_warp_knot,
            ),
            TimeWarp=dict(
                prob=self.time_warp_prob,
                sigma=self.time_warp_sigma,
                knot=self.time_warp_knot,
            ),
            WindowSlice=dict(
                prob=self.window_slice_prob, reduce_ratio=self.window_slice_reduce_ratio
            ),
            WindowWarp=dict(
                prob=self.window_warp_prob,
                window_ratio=self.window_warp_window_ratio,
                warp_slices=self.window_warp_scales,
            ),
        )
        for aug, params in aug_probs.items():
            if params["prob"] > 0:
                if aug == "Jitter":
                    self.transforms.append(Jitter(params["prob"], params["sigma"]))
                elif aug == "Scaling":
                    self.transforms.append(Scaling(params["prob"], params["sigma"]))
                elif aug == "Rotation":
                    self.transforms.append(Rotation(params["prob"]))
                elif aug == "Permutation":
                    self.transforms.append(
                        Permutation(
                            params["prob"], params["max_segments"], params["seg_mode"]
                        )
                    )
                elif aug == "MagnitudeWarp":
                    self.transforms.append(
                        MagnitudeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "TimeWarp":
                    self.transforms.append(
                        TimeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "WindowSlice":
                    self.transforms.append(
                        WindowSlice(params["prob"], params["reduce_ratio"])
                    )
                elif aug == "WindowWarp":
                    self.transforms.append(
                        WindowWarp(
                            params["prob"],
                            params["window_ratio"],
                            params["warp_slices"],
                        )
                    )

        self.augmentations = ApplyAugmentations(self.transforms)

    # greedy prediction
    def forward(self, *args, **kwargs):
        past_target = kwargs[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = kwargs[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        if self.time_feat:
            past_time_feat = kwargs["past_time_feat"]
            future_time_feat = kwargs["future_time_feat"]

        use_single_pass_sampling = self.use_single_pass_sampling

        future_samples = []

        if use_single_pass_sampling:
            # Single-pass sampling mode: Single forward pass per step, save distribution parameters, sample `num_parallel_samples` times, add mean to context.
            for t in range(self.prediction_length):
                params, loc, scale = self.model(
                    *args,
                    past_time_feat=past_time_feat if self.time_feat else None,
                    future_time_feat=future_time_feat[..., : t + 1, :]
                    if self.time_feat
                    else None,
                    past_target=past_target,
                    past_observed_values=past_observed_values,
                    use_kv_cache=self.use_kv_cache,
                )

                sliced_params = [
                    p[:, -1:] for p in params
                ]  # Take the last timestep predicted. Each tensor is of shape (#bsz, 1)
                # Singular distribution is used for getting the greedy prediction (mean)
                distr = self.model.distr_output.distribution(sliced_params, loc, scale)
                greedy_prediction = distr.mean  # (#bsz, 1)

                repeated_sliced_params = [
                    p[:, -1:].repeat_interleave(self.model.num_parallel_samples, 0)
                    for p in params
                ]  # Take the last timestep predicted and repeat for number of samples. Each tensor is of shape (#bsz*#parallel_samples, 1)
                repeated_loc = loc.repeat_interleave(self.model.num_parallel_samples, 0)
                repeated_scale = scale.repeat_interleave(
                    self.model.num_parallel_samples, 0
                )
                # Repeated distribution is used for getting the parallel samples
                # (distr.sample([self.model.num_parallel_samples]) seems to give terrible results)
                repeated_distr = self.model.distr_output.distribution(
                    repeated_sliced_params, repeated_loc, repeated_scale
                )
                sample = repeated_distr.sample()  # (#bsz*#parallel_samples, 1)
                if self.nonnegative_pred_samples:
                    sample = F.relu(sample)
                future_samples.append(sample)

                past_target = torch.cat((past_target, greedy_prediction), dim=1)
                past_observed_values = torch.cat(
                    (past_observed_values, torch.ones_like(greedy_prediction)), dim=1
                )
        else:
            # Original probabilistic forecasting: Duplicate input, `num_parallel_samples` forward passes per step, sample each distribution once, add samples to context.
            repeated_past_target = past_target.repeat_interleave(
                self.model.num_parallel_samples, 0
            )
            repeated_past_observed_values = past_observed_values.repeat_interleave(
                self.model.num_parallel_samples, 0
            )
            if self.time_feat:
                repeated_past_time_feat = past_time_feat.repeat_interleave(
                    self.model.num_parallel_samples, 0
                )
                repeated_future_time_feat = future_time_feat.repeat_interleave(
                    self.model.num_parallel_samples, 0
                )

            for t in range(self.prediction_length):
                if self.time_feat:
                    params, loc, scale = self.model(
                        *args,
                        past_time_feat=repeated_past_time_feat,
                        future_time_feat=repeated_future_time_feat[..., : t + 1, :],
                        past_target=repeated_past_target,
                        past_observed_values=repeated_past_observed_values,
                        use_kv_cache=self.use_kv_cache,
                    )
                else:
                    params, loc, scale = self.model(
                        *args,
                        past_time_feat=None,
                        future_time_feat=None,
                        past_target=repeated_past_target,
                        past_observed_values=repeated_past_observed_values,
                        use_kv_cache=self.use_kv_cache,
                    )

                sliced_params = [p[:, -1:] for p in params]
                distr = self.model.distr_output.distribution(sliced_params, loc, scale)
                sample = distr.sample()
                if self.nonnegative_pred_samples:
                    sample = F.relu(sample)
                future_samples.append(sample)

                repeated_past_target = torch.cat((repeated_past_target, sample), dim=1)
                repeated_past_observed_values = torch.cat(
                    (repeated_past_observed_values, torch.ones_like(sample)), dim=1
                )

        self.model.reset_cache()

        concat_future_samples = torch.cat(future_samples, dim=-1)
        return concat_future_samples.reshape(
            (-1, self.model.num_parallel_samples, self.prediction_length)
            + self.model.distr_output.event_shape,
        )

    # train
    def _compute_loss(self, batch, do_not_average=False, return_observed_values=False):
        past_target = batch[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = batch[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq)) with 0s or 1s indicating available (1s) or missing (0s)
        future_target = batch["future_target"]  # (bsz, model.prediction_length)
        future_observed_values = batch[
            "future_observed_values"
        ]  # (bsz, model.prediction_length) with 0s or 1s indicating available (1s) or missing (0s)
        if self.time_feat:
            past_time_feat = batch["past_time_feat"]
            future_time_feat = batch["future_time_feat"]
        else:
            past_time_feat = None
            future_time_feat = None

        extra_dims = len(future_target.shape) - len(past_target.shape)  # usually 0
        extra_shape = future_target.shape[:extra_dims]  # shape remains the same

        repeats = prod(extra_shape)  # usually 1
        past_target = repeat_along_dim(
            past_target, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = repeat_along_dim(
            past_observed_values, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)

        distr_args, loc, scale = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target_reshaped,
        )  # distr_args is a tuple with two tensors of shape (bsz, context_length+pred_len-1)
        context_target = take_last(
            past_target, dim=-1, num=self.context_length - 1
        )  # (bsz, context_length-1) # Basically removes the first value since it cannot be predicted
        target = torch.cat(
            (context_target, future_target_reshaped),
            dim=1,
        )  # (bsz, context_length-1+pred_len) # values that can be predicted
        context_observed = take_last(
            past_observed_values, dim=-1, num=self.context_length - 1
        )  # same as context_target, but for observed_values tensor
        observed_values = torch.cat(
            (context_observed, future_observed_reshaped), dim=1
        )  # same as target but for observed_values tensor

        if isinstance(self.model.distr_output, ImplicitQuantileNetworkOutput):
            if not do_not_average:
                loss = (
                    self.model.distr_output.loss(target, distr_args, loc, scale)
                    * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss = (
                    self.model.distr_output.loss(target, distr_args, loc, scale)
                    * observed_values
                )
        else:
            distr = self.model.distr_output.distribution(
                distr_args, loc=loc, scale=scale
            )  # an object representing a distribution with the specified parameters. We need this to compute the NLL loss.
            if not do_not_average:
                loss = (
                    self.loss(distr, target) * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss = self.loss(distr, target) * observed_values

        if not return_observed_values:
            return loss
        else:
            return loss, observed_values

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        if random.random() < self.aug_prob:
            # Freq mix and Freq mask have separate functions
            if self.freq_mask_rate > 0:
                batch["past_target"], batch["future_target"] = freq_mask(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mask_rate,
                )
            if self.freq_mixing_rate:
                batch["past_target"], batch["future_target"] = freq_mix(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mixing_rate,
                )
            # Other augmentation
            if len(self.transforms):
                batch["past_target"], batch["future_target"] = self.augmentations(
                    batch["past_target"], batch["future_target"]
                )

        train_loss_per_sample, observed_values = self._compute_loss(
            batch, do_not_average=True, return_observed_values=True
        )

        train_loss_avg = train_loss_per_sample.sum() / observed_values.sum().clamp_min(
            1.0
        )
        self.log(
            "train_loss", train_loss_avg, on_epoch=True, on_step=False, prog_bar=False
        )
        return train_loss_avg

    def on_train_epoch_end(self):
        # Log all losses
        for key, value in self.train_loss_dict.items():
            loss_avg = np.mean(value)
            self.log(
                f"train_loss_avg_per_train_dataset/{self.data_id_to_name_map[key]}",
                loss_avg,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )

        if self.track_loss_per_series:
            # Log all losses
            for key, value in self.train_loss_dict_per_series.items():
                loss_avg = np.mean(value)
                self.log(
                    f"train_loss_avg_per_train_series/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        # Reset loss_dict
        self.train_loss_dict = {}
        self.train_loss_dict_per_series = {}

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss_per_sample, observed_values = self._compute_loss(
            batch, do_not_average=True, return_observed_values=True
        )

        val_loss_avg = val_loss_per_sample.sum() / observed_values.sum().clamp_min(1.0)
        self.log("val_loss", val_loss_avg, on_epoch=True, on_step=False, prog_bar=False)
        return val_loss_avg

    def on_validation_epoch_end(self):
        # Log all losses
        for key, value in self.val_loss_dict.items():
            loss_avg = np.mean(value)
            if key >= 0:
                self.log(
                    f"val_loss_avg_per_train_dataset/{self.data_id_to_name_map[key]}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
            else:
                self.log(
                    f"val_loss_avg_per_test_dataset/{self.data_id_to_name_map[key]}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        if self.track_loss_per_series:
            # Log all losses
            for key, value in self.val_loss_dict_per_series.items():
                loss_avg = np.mean(value)
                self.log(
                    f"val_loss_avg_per_train_series/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        # Reset loss_dict
        self.val_loss_dict = {}
        self.val_loss_dict_per_series = {}

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.use_cosine_annealing_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.cosine_annealing_lr_args, verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
