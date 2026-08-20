# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements PatchTSMixer for forecasting."""

__author__ = ["Faakhir30"]
__all__ = ["PatchTSMixerForecaster"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
PatchTSMixerConfig = _safe_import("transformers.PatchTSMixerConfig")
PatchTSMixerForPrediction = _safe_import("transformers.PatchTSMixerForPrediction")
Trainer = _safe_import("transformers.Trainer")
TrainingArguments = _safe_import("transformers.TrainingArguments")

TIMESTAMP_COLUMN = "__PATCHTSM_TIMESTAMP_COL__"

_DEFAULT_CONFIG = {
    "context_length": 512,
    "prediction_length": 96,
    "patch_length": 8,
    "patch_stride": 8,
    "scaling": "std",
    "training_output_dir": "patchtsmixer_output",
    "training_label_names": ["future_values"],
    "training_report_to": "none",
}


class PatchTSMixerForecaster(BaseForecaster):
    """Forecaster wrapping IBM PatchTSMixer (granite-tsfm / Hugging Face).

    PatchTSMixer, developed by IBM, is a Lightweight MLP-Mixer Model for Multivariate
    Time Series Forecasting. Implementation inspired by [1]_.

    ``y`` should use a ``DatetimeIndex`` (or ``PeriodIndex``). Endogenous columns in
    ``y`` are forecast jointly; there is no exogenous ``X`` support. If ``y`` has no
    time index, index is reset, and a synthetic daily timestamp column is used instead.

    Parameters
    ----------
    model_path : str, optional, default="ibm-granite/granite-timeseries-patchtsmixer"
        Hugging Face model id or local checkpoint path. If ``None``, the model is
        initialized from ``config`` only (train from scratch).
    revision : str, default="main"
        Hub revision for ``from_pretrained``.
    config : dict, optional, default=None
        Extra fields for ``PatchTSMixerConfig`` (e.g. ``d_model``,
        ``patch_length``). Valid keys include:

        context_length : int, optional, default=32
            The context/history length for the input sequence.
        patch_length : int, optional, default=8
            The patch length for the input sequence.
        num_input_channels : int, optional, default=1
            The number of input channels.
        patch_stride : int, optional, default=8
            Determines the overlap between two consecutive patches. Set it to
            ``patch_length`` (or greater) for non-overlapping patches.
        num_parallel_samples : int, optional, default=100
            The number of samples to generate in parallel for probabilistic
            forecast.
        d_model : int, optional, default=8
            Size of the encoder layers and the pooler layer.
        expansion_factor : int, optional, default=2
            Expansion factor to use inside MLP. Recommended range is 2-5.
            Larger value indicates more complex model.
        num_layers : int, optional, default=3
            Number of hidden layers in the Transformer decoder.
        dropout : float or int, optional, default=0.2
            The ratio for all dropout layers.
        mode : str, optional, default="common_channel"
            Mixer mode. Determines how to process the channels. Allowed
            values: ``"common_channel"``, ``"mix_channel"``. In
            ``"common_channel"`` mode, channel-independent modelling is used
            with no explicit channel-mixing; channel mixing happens
            implicitly via shared weights across channels (preferred first
            approach). In ``"mix_channel"`` mode, explicit channel-mixing is
            used in addition to patch and feature mixer (preferred when
            channel correlations are very important to model).
        gated_attn : bool, optional, default=True
            Enable gated attention.
        norm_mlp : str, optional, default="LayerNorm"
            Normalization layer (``BatchNorm`` or ``LayerNorm``).
        self_attn : bool, optional, default=False
            Enable tiny self-attention across patches. This can be enabled
            when the output of vanilla PatchTSMixer with gated attention is
            not satisfactory. Enabling this leads to explicit pair-wise
            attention and modelling across patches.
        self_attn_heads : int, optional, default=1
            Number of self-attention heads. Works only when ``self_attn`` is
            set to ``True``.
        use_positional_encoding : bool, optional, default=False
            Enable the use of positional embedding for the tiny self-attention
            layers. Works only when ``self_attn`` is set to ``True``.
        positional_encoding_type : str, optional, default="sincos"
            Positional encodings. Options ``"random"`` and ``"sincos"`` are
            supported. Works only when ``use_positional_encoding`` is set to
            ``True``.
        scaling : str or bool, optional, default="std"
            Whether to scale the input targets via ``"mean"`` scaler,
            ``"std"`` scaler or no scaler if ``None``. If ``True``, the scaler
            is set to ``"mean"``.
        loss : str, optional, default="mse"
            The loss function for the model corresponding to the
            ``distribution_output`` head. For parametric distributions it is
            the negative log likelihood (``"nll"``) and for point estimates it
            is the mean squared error ``"mse"``.
        init_std : float, optional, default=0.02
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        norm_eps : float, optional, default=1e-05
            A value added to the denominator for numerical stability of
            normalization.
        mask_type : str, optional, default="random"
            Type of masking to use for masked pretraining mode. Allowed
            values are ``"random"``, ``"forecast"``. In random masking, points
            are masked randomly. In forecast masking, points are masked
            towards the end.
        random_mask_ratio : float, optional, default=0.5
            Masking ratio to use when ``mask_type`` is ``"random"``. Higher
            value indicates more masking.
        num_forecast_mask_patches : int or list, optional, default=[2]
            Number of patches to be masked at the end of each batch sample.
            If it is an integer, all the samples in the batch will have the
            same number of masked patches. If it is a list, samples in the
            batch will be randomly masked by numbers defined in the list.
            This argument is only used for forecast pretraining.
        mask_value : float, optional, default=0.0
            Mask value to use.
        masked_loss : bool, optional, default=True
            Whether to compute pretraining loss only at the masked portions,
            or on the entire output.
        channel_consistent_masking : bool, optional, default=True
            When ``True``, masking will be the same across all channels of a
            timeseries. Otherwise, masking positions will vary across
            channels.
        unmasked_channel_indices : list, optional
            Channels that are not masked during pretraining.
        head_dropout : float, optional, default=0.2
            The dropout probability for the PatchTSMixer head.
        distribution_output : str, optional, default="student_t"
            The distribution emission head for the model when ``loss`` is
            ``"nll"``. Could be either ``"student_t"``, ``"normal"`` or
            ``"negative_binomial"``.
        prediction_length : int, optional, default=16
            Number of time steps to forecast for a forecasting task. Also
            known as the forecast horizon.
        prediction_channel_indices : list, optional
            List of channel indices to forecast. If ``None``, forecast all
            channels. Target data is expected to have all channels and we
            explicitly filter the channels in prediction and target before
            loss computation.
        num_targets : int, optional, default=3
            Number of targets (dimensionality of the regressed variable) for
            a regression task.
        output_range : list, optional
            Output range to restrict for the regression task. Defaults to
            ``None``.
        head_aggregation : str, optional, default="max_pool"
            Aggregation mode to enable for classification or regression task.
            Allowed values are ``None``, ``"use_last"``, ``"max_pool"``,
            ``"avg_pool"``.

    context_length : int, optional, default=None
        Input history length for sliding windows. If ``None``, taken from the loaded
        config or defaults to ``512`` when training from scratch.
    prediction_length : int, optional, default=None
        Forecast horizon length for the model head. If ``None``, uses ``max(fh)`` when
        ``fh`` is passed to ``fit``, else the loaded config default.
    validation_split : float, optional, default=0.2
        Fraction of ``y`` held out for validation during ``Trainer`` training.
    train_model : bool, default=True
        If ``True``, run ``Trainer.train()`` on ``y``. If ``False``, only fit the
        preprocessor and load weights (pretrained model evaluate path).
    scaling : bool, default=True
        Whether ``TimeSeriesPreprocessor`` standardizes targets.
    training_args : dict, optional, default=None
        Passed to ``TrainingArguments`` (``label_names=["future_values"]`` is set if
        missing). See [3]_ for details.
    callbacks : list, optional, default=None
        Hugging Face ``Trainer`` callbacks (e.g. ``EarlyStoppingCallback``).
    num_parallel_samples : int, optional, default=None
        Override ``num_parallel_samples`` on the model for ``generate``.

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/patch_tsmixer_getting_started.ipynb
    .. [2] Ekambaram et al., TSMixer: Lightweight MLP-Mixer Model for Multivariate
           Time Series Forecasting, arXiv:2306.09364
    .. [3] https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.patch_tsmixer import PatchTSMixerForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, _ = temporal_train_test_split(y)
    >>> f = PatchTSMixerForecaster(  # doctest: +SKIP
    ...     model_path=None,
    ...     config={
    ...         "context_length": 8,
    ...         "prediction_length": 3,
    ...         "patch_length": 2,
    ...         "patch_stride": 2,
    ...         "num_input_channels": 1,
    ...         "d_model": 16,
    ...         "num_layers": 1,
    ...     },
    ...     training_args={
    ...         "output_dir": "test_output",
    ...         "max_steps": 2,
    ...         "per_device_train_batch_size": 4,
    ...         "report_to": "none",
    ...     },
    ... )
    >>> f.fit(y_train, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = f.predict()  # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "Faakhir30",
            # IBM authors:
            "ajati",
            "vijaye12ibm",
            "Phanwadee Sinthong",
            "Nam Nguyen",
            "Jayant Kalagnanam",
        ],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.11",
        "python_dependencies": [
            "granite-tsfm>=0.3.5",
            "torch",
            "transformers",
            "accelerate",
        ],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": False,
        "requires-fh-in-fit": False,
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str | None = "ibm-granite/granite-timeseries-patchtsmixer",
        revision: str = "main",
        config: dict | None = None,
        context_length: int | None = None,
        prediction_length: int | None = None,
        validation_split: float = 0.2,
        train_model: bool = True,
        scaling: bool = True,
        training_args: dict | None = None,
        callbacks: list | None = None,
        num_parallel_samples: int | None = None,
    ):
        self.model_path = model_path
        self.revision = revision
        self.config = config
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.validation_split = validation_split
        self.train_model = train_model
        self.scaling = scaling
        self.training_args = training_args
        self.callbacks = callbacks
        self.num_parallel_samples = num_parallel_samples
        self.model = None
        super().__init__()

    def __post_init__(self):
        """Post-initialization setup."""
        self._config = {} if self.config is None else self.config.copy()
        self._training_args = (
            {} if self.training_args is None else self.training_args.copy()
        )

    def _add_timestamp_col(self, y):
        """Add timestamp column to y and reset index."""
        df = y.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif isinstance(df.index, pd.PeriodIndex):
            timestamps = df.index.to_timestamp()
        else:
            timestamps = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
            df = df.reset_index(drop=True)

        df[TIMESTAMP_COLUMN] = timestamps

        return df

    def _prediction_length_from_fh(self, fh):
        if fh is None:
            return self.prediction_length
        fh_rel = fh.to_relative(self._cutoff)
        return int(np.max(fh_rel.to_numpy()))

    def _resolve_lengths(self, fh, n_channels):
        pred_len = self.prediction_length
        if pred_len is None and fh is not None:
            pred_len = self._prediction_length_from_fh(fh)
        if pred_len is None:
            pred_len = self._config.get(
                "prediction_length", _DEFAULT_CONFIG["prediction_length"]
            )

        ctx_len = self.context_length
        if ctx_len is None:
            ctx_len = self._config.get(
                "context_length", _DEFAULT_CONFIG["context_length"]
            )

        return ctx_len, pred_len, n_channels

    def _build_model_config(self, context_length, prediction_length, n_channels):
        cfg = dict(self._config)
        cfg.setdefault("context_length", context_length)
        cfg.setdefault("prediction_length", prediction_length)
        cfg.setdefault("num_input_channels", n_channels)
        cfg.setdefault(
            "patch_stride",
            cfg.get("patch_length", _DEFAULT_CONFIG["patch_length"]),
        )
        cfg.setdefault("scaling", _DEFAULT_CONFIG["scaling"])
        return cfg

    def _load_model(self, config):
        if self.model_path is None:
            return PatchTSMixerForPrediction(config=config)

        return PatchTSMixerForPrediction.from_pretrained(
            self.model_path,
            revision=self.revision,
            config=config,
            ignore_mismatched_sizes=True,
        )

    def _fit(self, y, X=None, fh=None):
        from tsfm_public.toolkit.dataset import ForecastDFDataset
        from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

        data = self._add_timestamp_col(y)
        target_cols = [col for col in data.columns if col != TIMESTAMP_COLUMN]
        n_channels = len(target_cols)
        context_length, prediction_length, _ = self._resolve_lengths(fh, n_channels)
        if fh is not None:
            prediction_length = max(
                prediction_length, self._prediction_length_from_fh(fh)
            )

        if self.validation_split and self.validation_split > 0:
            train_df, valid_df = temporal_train_test_split(
                data, test_size=self.validation_split
            )
        else:
            train_df, valid_df = data, None

        self._preprocessor = TimeSeriesPreprocessor(
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=target_cols,
            scaling=self.scaling,
        )
        self._preprocessor.train(train_df)

        self._timestamp_column = TIMESTAMP_COLUMN
        self._target_columns = target_cols
        self._context_length = context_length
        self._prediction_length = prediction_length

        train_dataset = ForecastDFDataset(
            self._preprocessor.preprocess(train_df),
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=target_cols,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        eval_dataset = None
        if valid_df is not None:
            eval_dataset = ForecastDFDataset(
                self._preprocessor.preprocess(valid_df),
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_cols,
                context_length=context_length,
                prediction_length=prediction_length,
            )

        if self.model_path is not None:
            hub_cfg = PatchTSMixerConfig.from_pretrained(
                self.model_path, revision=self.revision
            )
            merged = {
                **hub_cfg.to_dict(),
                **self._build_model_config(
                    context_length, prediction_length, n_channels
                ),
            }
            config = PatchTSMixerConfig(**merged)
        else:
            config = PatchTSMixerConfig(
                **self._build_model_config(
                    context_length, prediction_length, n_channels
                )
            )

        self.model = self._load_model(config)
        if self.num_parallel_samples is not None:
            self.model.num_parallel_samples = self.num_parallel_samples

        if not self.train_model:
            self.model.eval()
            return self

        training_args_dict = dict(self._training_args)
        training_args_dict.setdefault(
            "label_names", _DEFAULT_CONFIG["training_label_names"]
        )
        training_args_dict.setdefault(
            "output_dir", _DEFAULT_CONFIG["training_output_dir"]
        )
        training_args_dict.setdefault(
            "report_to", _DEFAULT_CONFIG["training_report_to"]
        )
        if not _check_soft_dependencies("transformers<5.0", severity="none"):
            training_args_dict.pop("overwrite_output_dir", None)

        train_args = TrainingArguments(**training_args_dict)
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=self.callbacks,
        )
        trainer.train()

        self.model = trainer.model
        self.model.eval()
        return self

    def _inference_batch(self, y):
        """Last sliding window from preprocessed history."""
        from tsfm_public.toolkit.dataset import ForecastDFDataset

        data = self._add_timestamp_col(y)
        target_cols = [col for col in data.columns if col != TIMESTAMP_COLUMN]
        preprocessed = self._preprocessor.preprocess(data)
        dataset = ForecastDFDataset(
            preprocessed,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=target_cols,
            context_length=self._context_length,
            prediction_length=self._prediction_length,
        )
        return dataset[len(dataset) - 1]

    def _forward_window(self, batch):
        past = (
            batch["past_values"]
            .unsqueeze(0)
            .to(dtype=self.model.dtype, device=self.model.device)
        )
        mask = batch.get("past_observed_mask")
        if mask is not None:
            mask = mask.unsqueeze(0).to(
                dtype=self.model.dtype, device=self.model.device
            )
        with torch.inference_mode():
            return self.model(
                past_values=past,
                observed_mask=mask,
                return_dict=True,
            )

    def _point_predictions(self, out):
        po = out.prediction_outputs
        if isinstance(po, tuple):
            dist = self.model.distribution_output.distribution(
                po, loc=out.loc, scale=out.scale
            )
            return dist.mean
        return po

    def _predict(self, fh, X=None):
        if fh is None:
            fh = self.fh
        fh_rel = fh.to_relative(self.cutoff)

        batch = self._inference_batch(self._y)
        out = self._forward_window(batch)
        pred = self._point_predictions(out).detach().cpu().numpy()[0]
        n_cols = len(self._target_columns)
        pred = pred[:, :n_cols]

        rel_idx = (fh_rel.to_numpy() - 1).astype(int)
        values = pred[rel_idx]

        index = fh.to_absolute(self._cutoff)._values
        pred_df = pd.DataFrame(values, index=index, columns=self._target_columns)
        pred_df.index.names = self._y.index.names
        return pred_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        base_train = {
            "output_dir": "test_output",
            "max_steps": 2,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "report_to": "none",
        }
        return [
            {
                "model_path": None,
                "context_length": 8,
                "validation_split": 0.2,
                "config": {
                    "patch_length": 2,
                    "patch_stride": 2,
                    "d_model": 16,
                    "num_layers": 1,
                    "loss": "mse",
                },
                "training_args": base_train,
            },
            {
                "model_path": None,
                "context_length": 6,
                "train_model": False,
                "validation_split": 0.0,
                "config": {
                    "patch_length": 2,
                    "patch_stride": 2,
                    "d_model": 8,
                    "num_layers": 1,
                    "loss": "mse",
                },
            },
        ]
