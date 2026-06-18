# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Sundial forecaster for ``sktime``."""

__author__ = ["WenWeiTHU", "geetu040"]
# WenWeiTHU for thuml/sundial-base-128m

__all__ = ["SundialForecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.singleton import _multiton


class SundialForecaster(BaseForecaster):
    """Sundial forecaster via Hugging Face ``transformers``.

    This forecaster wraps Sundial [1]_, [2]_, [3]_ and exposes forecasting
    through the ``sktime`` forecasting interface. Calling :meth:`fit` loads the
    model and stores the observed series as forecasting context. Calling
    :meth:`pretrain` fine-tunes the model on panel or hierarchical data through
    the Sundial forward loss.

    Sundial generates one or more sample paths. Point forecasts are computed as
    the empirical mean over generated samples. Quantile forecasts are computed
    as empirical quantiles over generated samples.

    Parameters
    ----------
    model_path : str, default="thuml/sundial-base-128m"
        Hugging Face repository identifier or local path to a Sundial
        checkpoint. If ``None``, a model is created from ``config`` with random
        weights. Use this path for tests or pretraining from scratch; the model
        should be pretrained before it is used for meaningful forecasting.
    config : SundialConfig or dict, optional (default=None)
        Model configuration used when ``model_path=None``. If provided as a
        ``dict``, it is converted with ``SundialConfig.from_dict``. If ``None``
        and ``model_path=None``, the default ``SundialConfig`` is used. A config
        without pretrained weights initializes random weights and should be
        followed by :meth:`pretrain` before forecasting.
    device_map : str, dict, int, or torch.device, default="cpu"
        Device placement following the ``transformers`` ``device_map`` naming
        convention, for example ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or
        ``"auto"``.
    dtype : torch.dtype or str, optional (default=None)
        Data type used for model loading, following the ``transformers``
        ``dtype`` convention, for example ``torch.float16``,
        ``torch.bfloat16``, or ``"auto"``.
    forward_kwargs : dict, optional (default=None)
        Additional keyword arguments forwarded to ``model.generate(...)`` during
        prediction. Sundial-specific options include arguments such as
        ``num_samples`` and ``revin``; standard generation options supported by
        ``transformers.GenerationMixin.generate`` may also be passed. See the
        Sundial model card [3]_ and Transformers generation docs [4]_ for
        details.
    deterministic : bool, default=False
        Whether predictions should reset the ``transformers`` random seed before
        generation.
    validation_split : float or None, default=0.2
        Fraction of data reserved for evaluation when :meth:`pretrain` is used.
        If ``None``, no evaluation dataset is created.
    training_args : dict, optional (default=None)
        Keyword arguments used to construct ``transformers.TrainingArguments``
        in :meth:`pretrain` [5]_.
    compute_metrics : callable or dict, optional (default=None)
        Metrics callback(s) passed to ``transformers.Trainer`` [5]_.
    callbacks : list, optional (default=None)
        Trainer callbacks passed to ``transformers.Trainer`` [5]_.

    References
    ----------
    .. [1] Sundial: A Family of Highly Capable Time Series Foundation Models:
       https://arxiv.org/abs/2502.00816
    .. [2] Sundial repository:
       https://github.com/thuml/Sundial
    .. [3] Sundial model card:
       https://huggingface.co/thuml/sundial-base-128m
    .. [4] Transformers `.generate()`:
       https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate
    .. [5] Trainer/TrainingArguments docs:
       https://huggingface.co/docs/transformers/en/main_classes/trainer

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> y = load_airline()
    >>> forecaster = SundialForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Running with explicit device, dtype, and sampling settings:

    >>> import torch
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     device_map="cuda",
    ...     dtype=torch.bfloat16,
    ...     forward_kwargs={"num_samples": 20},
    ...     deterministic=True,
    ... )
    >>> y_pred = forecaster.fit(y).predict(fh=[1, 2, 3])  # doctest: +SKIP

    Passing Sundial and Transformers generation options through
    ``forward_kwargs``:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     forward_kwargs={"num_samples": 20, "revin": False},
    ... )
    >>> y_pred = forecaster.fit(y).predict(fh=[1, 2, 3])  # doctest: +SKIP

    Quantile prediction from generated samples:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     forward_kwargs={"num_samples": 50},
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 2, 3],
    ...     alpha=[0.1, 0.5, 0.9],
    ... )

    Global training on panel or hierarchical data:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sundial import SundialForecaster
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> y_panel = _make_hierarchical(  # doctest: +SKIP
    ...     hierarchy_levels=(3,),
    ...     min_timepoints=128,
    ...     max_timepoints=400,
    ... )
    >>> y = load_airline()
    >>> forecaster = SundialForecaster(  # doctest: +SKIP
    ...     training_args={"output_dir": "sundial-output", "max_steps": 10},
    ... )
    >>> forecaster.pretrain(y_panel)  # doctest: +SKIP
    >>> y_pred = forecaster.fit(y).predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pretrain": True,
        "authors": ["WenWeiTHU", "geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]~=4.40.0"],
        "tests:vm": True,
        "tests:libs": ["sktime.libs.sundial"],
    }

    def __init__(
        self,
        model_path="thuml/sundial-base-128m",
        config=None,
        device_map="cpu",
        dtype=None,
        forward_kwargs=None,
        deterministic=False,
        validation_split=0.2,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
    ):
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.forward_kwargs = forward_kwargs
        self.deterministic = deterministic
        self.validation_split = validation_split
        self.training_args = training_args
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks

        super().__init__()

    def _pretrain(self, y, X=None, fh=None):
        """Pretrain Sundial on panel/global data."""
        self.model_ = self._load_model()

        input_token_len = self.model_.config.input_token_len
        output_token_len = self.model_.config.output_token_lens[-1]

        horizon_length = _get_horizon_length(fh, output_token_len)
        if horizon_length > output_token_len:
            raise ValueError(
                "Requested pretraining horizon exceeds Sundial model capacity: "
                f"max requested step={horizon_length}, "
                f"max output_token_lens={output_token_len}. "
                "Use a shorter fh or a model/config with a larger "
                "output_token_lens value."
            )

        from transformers import Trainer, TrainingArguments

        if self.validation_split is not None:
            y_train, y_eval = temporal_train_test_split(
                y, test_size=self.validation_split
            )
        else:
            y_train = y
            y_eval = None

        train = SundialPyTorchDataset(
            series_list=_prepare_series_list(y_train),
        )

        eval = None
        if self.validation_split is not None and len(_prepare_series_list(y_eval)) > 0:
            eval = SundialPyTorchDataset(
                series_list=_prepare_series_list(y_eval),
            )
        elif self.validation_split is not None:
            warn(
                "Skipping Sundial evaluation dataset creation: validation split "
                "does not contain any non-empty numeric series. Training "
                "continues without evaluation.",
                stacklevel=2,
            )

        data_collator = SundialDataCollator(
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            horizon_length=horizon_length,
        )

        training_args = (
            deepcopy(self.training_args) if self.training_args is not None else {}
        )
        training_args = TrainingArguments(**training_args)

        self.model_.train()
        trainer = Trainer(
            model=self.model_,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train,
            eval_dataset=eval,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )
        trainer.train()

        self.model_ = trainer.model

    def _fit(self, y, X=None, fh=None):
        """Load the model and store history for forecasting."""
        self.model_ = self._load_model()
        self.context_ = y

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon."""
        self.model_.eval()
        samples, fh, preds_idx = self._generate_samples(fh)

        preds = samples.mean(axis=1)
        preds = preds[:, preds_idx].T
        preds = pd.DataFrame(
            preds,
            index=fh.to_absolute(self._cutoff)._values,
            columns=self.context_.columns,
        )

        return preds

    def _predict_quantiles(self, fh, X, alpha):
        """Compute empirical prediction quantiles from generated samples."""
        samples, fh, preds_idx = self._generate_samples(fh)

        if alpha is None:
            alpha = [0.1, 0.5, 0.9]
        alpha = [round(i, 3) for i in alpha]

        preds = np.quantile(samples, q=alpha, axis=1)
        preds = np.moveaxis(preds, 0, -1)
        preds = preds[:, preds_idx, :]
        preds = preds.transpose(1, 0, 2).reshape(len(preds_idx), -1)

        columns = pd.MultiIndex.from_product([self.context_.columns, alpha])
        pred_quantiles = pd.DataFrame(
            data=preds,
            index=fh.to_absolute(self._cutoff)._values,
            columns=columns,
        )

        return pred_quantiles

    def _generate_samples(self, fh):
        """Generate Sundial sample paths for the requested horizon."""
        import torch
        import transformers

        if self.deterministic:
            transformers.set_seed(42)

        self.model_ = self._load_model()

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        horizon_length = np.max(preds_idx) + 1
        output_token_lens = getattr(self.model_.config, "output_token_lens", [])
        if output_token_lens and max(output_token_lens) < horizon_length:
            raise ValueError(
                "Requested forecasting horizon exceeds Sundial model capacity: "
                f"max requested step={horizon_length}, "
                f"max output_token_lens={max(output_token_lens)}. "
                "Use a shorter horizon or a model/config with a larger "
                "output_token_lens value."
            )

        past_values = self.context_.to_numpy().T
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        forward_kwargs = {} if not self.forward_kwargs else self.forward_kwargs
        output = self.model_.generate(
            past_values,
            max_new_tokens=horizon_length,
            **forward_kwargs,
        )

        samples = output.detach().float().cpu().numpy()

        return samples, fh, preds_idx

    def _load_model(self):
        """Load or retrieve a cached Sundial model instance."""
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        self.model_ = _CachedSundial(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device_map=self.device_map,
            dtype=self.dtype,
        ).load()

        return self.model_

    def _get_unique_key(self):
        """Build cache key for the multiton model loader."""
        key = {
            "model_path": self.model_path,
            "config": self.config,
            "device_map": self.device_map,
            "dtype": self.dtype,
        }
        return str(sorted(key.items()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        config = {
            "input_token_len": 2,
            "hidden_size": 4,
            "intermediate_size": 8,
            "output_token_lens": [8],
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "max_position_embeddings": 64,
            "flow_loss_depth": 1,
            "num_sampling_steps": 1,
            "diffusion_batch_mul": 1,
            "use_cache": False,
        }
        return [
            {
                "model_path": None,
                "config": config,
                "device_map": "cpu",
                "forward_kwargs": {"num_samples": 2, "revin": True},
                "deterministic": True,
                "validation_split": 0.1,
                "training_args": {
                    "output_dir": "test_output",
                    "max_steps": 1,
                },
            },
            {
                "model_path": None,
                "config": config,
                "device_map": None,
                "forward_kwargs": {"num_samples": 3, "revin": False},
                "deterministic": True,
                "validation_split": 0.1,
                "training_args": {
                    "output_dir": "test_output",
                    "max_steps": 1,
                },
            },
        ]


@_multiton
class _CachedSundial:
    """Multiton-backed cache wrapper for a loaded Sundial model."""

    def __init__(
        self,
        key,
        model_path,
        config,
        device_map,
        dtype,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.model_ = None

    def load(self):
        """Load model if needed and return cached instance."""
        if self.model_ is not None:
            return self.model_

        if self.model_path is not None:
            self.model_ = self._load_from_path()
        else:
            self.model_ = self._load_randomly()

        return self.model_

    def _load_from_path(self):
        """Load Sundial model weights from ``self.model_path``."""
        from sktime.libs.sundial import SundialConfig, SundialForPrediction

        kwargs = {}
        if self.config is not None:
            config = deepcopy(self.config)
            if isinstance(config, dict):
                config = SundialConfig.from_dict(config)
            kwargs["config"] = config
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype

        return SundialForPrediction.from_pretrained(self.model_path, **kwargs)

    def _load_randomly(self):
        """Initialize a Sundial model randomly from config."""
        from sktime.libs.sundial import SundialConfig, SundialForPrediction

        warn(
            "Initializing Sundial from config creates random weights. Sundial "
            "pretraining is required before these weights are suitable for "
            "meaningful forecasting.",
            UserWarning,
            stacklevel=2,
        )

        config = deepcopy(self.config)
        if not config:
            config = SundialConfig()
        if isinstance(config, dict):
            config = SundialConfig.from_dict(config)

        model = SundialForPrediction(config)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)
        if self.device_map is not None:
            model = model.to(self.device_map)

        return model


def _get_horizon_length(fh, output_token_len):
    """Return pretraining horizon length from fh or model output capacity."""
    if fh is None:
        return output_token_len
    fh = fh.to_relative(cutoff=0)
    return int(np.max(fh._values.values))


def _prepare_series_list(data):
    """Convert panel/hierarchical DataFrame into 1D training series."""
    if data.index.nlevels == 1:
        groups = [(None, data)]
    else:
        instance_levels = list(range(data.index.nlevels - 1))
        groupby_level = (
            instance_levels[0] if len(instance_levels) == 1 else instance_levels
        )
        groups = data.groupby(level=groupby_level)

    series_list = []
    for _, group in groups:
        for col in group.columns:
            values = group[col].to_numpy(dtype=np.float32)
            values = values[np.isfinite(values)]
            series_list.append(values)

    return series_list


class SundialPyTorchDataset:
    """Dataset for Sundial pretraining with ``Trainer``."""

    def __init__(self, series_list):
        self.series_list = [series for series in series_list if len(series) > 0]

        if not self.series_list:
            raise ValueError(
                "No training series were available for Sundial pretraining. "
                "Provide at least one non-empty numeric series."
            )

    def __len__(self):
        """Return number of training series."""
        return len(self.series_list)

    def __getitem__(self, i):
        """Return one unpadded training series."""
        import torch

        return torch.tensor(self.series_list[i], dtype=torch.float32)


class SundialDataCollator:
    """Pad Sundial training series to a common batch shape."""

    def __init__(self, input_token_len, output_token_len, horizon_length):
        self.input_token_len = input_token_len
        self.output_token_len = output_token_len
        self.horizon_length = horizon_length

    def __call__(self, features):
        """Collate variable-length series into Sundial training tensors."""
        import torch

        lengths = [len(feature) for feature in features]
        max_length = max(lengths)
        input_length = (
            (max_length + self.input_token_len - 1) // self.input_token_len
        ) * self.input_token_len
        label_length = input_length - self.input_token_len + self.output_token_len
        n_input_tokens = input_length // self.input_token_len

        input_ids = []
        labels = []
        loss_masks = []
        for feature, length in zip(features, lengths):
            left_pad = input_length - length
            label_right_pad = label_length - left_pad - length

            input_ids.append(torch.nn.functional.pad(feature, (left_pad, 0), value=0.0))
            labels.append(
                torch.nn.functional.pad(
                    feature,
                    (left_pad, label_right_pad),
                    value=0.0,
                )
            )

            token_starts = np.arange(n_input_tokens) * self.input_token_len
            token_starts = token_starts - left_pad
            valid_targets = token_starts + self.horizon_length <= length
            valid_targets = valid_targets & (token_starts >= 0)
            loss_masks.append(torch.tensor(valid_targets, dtype=torch.float32))

        mask_y = torch.zeros(len(features), self.output_token_len, dtype=torch.float32)
        mask_y[:, : self.horizon_length] = 1.0

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "loss_masks": torch.stack(loss_masks),
            "mask_y": mask_y,
        }
