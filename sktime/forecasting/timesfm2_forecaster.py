# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TimesFM-2 forecaster for ``sktime``.

This module provides an ``sktime`` forecaster wrapping the Hugging Face
``transformers`` implementation of Google Research TimesFM-2 models
(TimesFM-2.0 and TimesFM-2.5). It supports:

- zero-shot prediction through :meth:`fit` + :meth:`predict`
- global pretraining on panel/hierarchical data through :meth:`pretrain`
- quantile prediction through :meth:`predict_quantiles`
- Hugging Face ``device_map`` and ``dtype`` model-loading options
- optional quantized pretrained model loading
- optional PEFT wrapping for pretrained models
"""

__author__ = ["rajatsen91", "siriuz42", "geetu040"]
# rajatsen91 for google-research/timesfm

__all__ = ["TimesFM2Forecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.singleton import _multiton


class TimesFM2Forecaster(BaseForecaster):
    """TimesFM-2.x forecaster via Hugging Face ``transformers``.

    This forecaster wraps TimesFM-2 prediction models [1]_, [2]_ from Hugging
    Face and exposes them through the ``sktime`` forecasting interface.

    Two primary workflows are supported:

    1. ``fit`` for zero-shot inference setup (loads model and stores history).
    2. ``pretrain`` for global fine-tuning on panel/hierarchical data.

    Notes
    -----
    - Prediction is bounded by ``model.config.horizon_length``. Requested
      forecast steps beyond this limit raise ``ValueError``.
    - Quantile prediction is only available for quantiles present in
      ``model.config.quantiles``.
    - ``device_map`` and ``dtype`` are supported for both pretrained loading
      and config-only initialization. For pretrained loading, they are passed
      to ``from_pretrained``; for config-only initialization, they are applied
      after model construction.
    - ``quantization_config`` and ``peft_config`` are applied only when loading
      a pretrained model from ``model_path``. They are not used for config-only
      initialization with ``model_path=None``.
    - Loaded models are cached via a multiton helper keyed by model-loading
      inputs to avoid repeated model instantiation.

    Parameters
    ----------
    model_path : str, default="google/timesfm-2.5-200m-transformers"
        Hugging Face repository identifier or local path to a TimesFM checkpoint.
        Defaults to the TimesFM-2.5 checkpoint [3]_; TimesFM-2.0 checkpoints
        are also supported [4]_. If ``None``, a model is created from
        ``config`` (or default ``transformers.TimesFmConfig``).
    config : transformers.PretrainedConfig or dict, optional (default=None)
        Model configuration used for loading/initialization.

        - If ``model_path`` is not ``None``:
          passed to ``from_pretrained(..., config=config)``.
        - If ``model_path`` is ``None``:
          used to instantiate a model from configuration.

        If provided as ``dict``, the architecture entry (for example
        ``"TimesFmModelForPrediction"`` or ``"TimesFm2_5ModelForPrediction"``)
        is used to infer the config class.
    device_map : str, dict, int, or torch.device, default="cpu"
        Device placement following the ``transformers`` ``device_map`` naming
        convention, for example ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or
        ``"auto"``.
    dtype : torch.dtype or str, optional (default=None)
        Data type used for model loading, following the ``transformers``
        ``dtype`` convention, for example ``torch.float16``,
        ``torch.bfloat16``, or ``"auto"``.
    quantization_config : transformers.quantizers.HfQuantizer, optional
        Valid quantization configuration object compatible with pretrained
        loading through ``transformers.PreTrainedModel.from_pretrained`` [8]_.
        Applied only when ``model_path`` is not ``None``; ignored for
        config-only initialization with ``model_path=None``.
    forward_kwargs : dict, optional (default=None)
        Additional keyword arguments forwarded to ``model(...)`` during
        :meth:`predict` and :meth:`predict_quantiles`; see the TimesFM-2.0 [5]_
        and TimesFM-2.5 [6]_ forward APIs.
    peft_config : peft.PeftConfig, optional (default=None)
        If provided, wraps the loaded pretrained base model with PEFT using
        ``peft.get_peft_model``. Applied only when ``model_path`` is not
        ``None``; ignored for config-only initialization with
        ``model_path=None``.
    validation_split : float or None, default=0.2
        Fraction of data reserved for evaluation when :meth:`pretrain` is used.
        If ``None``, no evaluation dataset is created.
    training_args : dict, optional (default=None)
        Keyword arguments used to construct ``transformers.TrainingArguments``
        in :meth:`pretrain` [7]_.
    compute_loss_func : callable, optional (default=None)
        Optional custom loss function passed to ``transformers.Trainer`` [7]_.
    compute_metrics : callable or dict, optional (default=None)
        Metrics callback(s) passed to ``transformers.Trainer`` [7]_.
    callbacks : list, optional (default=None)
        Trainer callbacks passed to ``transformers.Trainer`` [7]_.

    References
    ----------
    .. [1] Das, A., Kong, W., Sen, R., and Zhou, Y. (2024).
       A Decoder-only Foundation Model for Time-series Forecasting.
       CoRR. https://arxiv.org/abs/2310.10688
    .. [2] Google Research TimesFM repository:
       https://github.com/google-research/timesfm
    .. [3] TimesFM-2.5 model card:
       https://huggingface.co/google/timesfm-2.5-200m-transformers
    .. [4] TimesFM-2.0 model card:
       https://huggingface.co/google/timesfm-2.0-500m-pytorch
    .. [5] TimesFM-2.0 forward API:
       https://huggingface.co/docs/transformers/en/model_doc/timesfm#transformers.TimesFmModelForPrediction.forward
    .. [6] TimesFM-2.5 forward API:
       https://huggingface.co/docs/transformers/en/model_doc/timesfm2_5#transformers.TimesFm2_5ModelForPrediction.forward
    .. [7] Trainer/TrainingArguments docs:
       https://huggingface.co/docs/transformers/en/main_classes/trainer
    .. [8] Quantization docs:
       https://huggingface.co/docs/transformers/en/main_classes/quantization

    Examples
    --------
    Simple zero-shot forecasting with TimesFM-2.5:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm2_forecaster import TimesFM2Forecaster
    >>> y = load_airline()
    >>> # By default, loads google/timesfm-2.5-200m-transformers.
    >>> forecaster = TimesFM2Forecaster()  # doctest: +SKIP
    >>> # fit loads the model weights and stores the forecasting context.
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Simple zero-shot forecasting with TimesFM-2.0:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm2_forecaster import TimesFM2Forecaster
    >>> y = load_airline()
    >>> # Loads google/timesfm-2.0-500m-pytorch.
    >>> forecaster = TimesFM2Forecaster(  # doctest: +SKIP
    ...     model_path="google/timesfm-2.0-500m-pytorch",
    ...     forward_kwargs={"forecast_context_len": 1024},
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Quantile prediction:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm2_forecaster import TimesFM2Forecaster
    >>> y = load_airline()
    >>> forecaster = TimesFM2Forecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> # Select only quantiles available in the model config.
    >>> y_pred = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 2, 3],
    ...     alpha=[0.1, 0.5, 0.9],
    ... )

    Reduced-memory inference with device placement, dtype, and quantization:

    >>> import torch  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm2_forecaster import TimesFM2Forecaster
    >>> from transformers import QuantoConfig  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = TimesFM2Forecaster(  # doctest: +SKIP
    ...     model_path="google/timesfm-2.5-200m-transformers",
    ...     device_map="auto",
    ...     dtype=torch.bfloat16,
    ...     quantization_config=QuantoConfig(weights="int8"),
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Global training with a PEFT-wrapped pretrained model:

    >>> from peft import LoraConfig  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm2_forecaster import TimesFM2Forecaster
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> y_panel = _make_hierarchical(  # doctest: +SKIP
    ...     hierarchy_levels=(3,),
    ...     min_timepoints=128,
    ...     max_timepoints=400,
    ... )
    >>> y = load_airline()
    >>> forecaster = TimesFM2Forecaster(  # doctest: +SKIP
    ...     model_path="google/timesfm-2.5-200m-transformers",
    ...     peft_config=LoraConfig(
    ...         r=8,
    ...         lora_alpha=32,
    ...         target_modules=["q_proj", "v_proj"],
    ...         lora_dropout=0.01,
    ...     ),
    ... )
    >>> # Training happens on hierarchical data.
    >>> forecaster.pretrain(y_panel)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Global training on a randomly initialized model with custom config:
    ``device_map`` and ``dtype`` can still be applied in this path, but
    ``quantization_config`` and ``peft_config`` require a pretrained
    ``model_path`` and are ignored when ``model_path=None``.

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm2_forecaster import TimesFM2Forecaster
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> y_panel = _make_hierarchical(  # doctest: +SKIP
    ...     hierarchy_levels=(3,),
    ...     min_timepoints=128,
    ...     max_timepoints=400,
    ... )
    >>> y = load_airline()
    >>> forecaster = TimesFM2Forecaster(  # doctest: +SKIP
    ...     model_path=None,
    ...     config={
    ...         "architectures": ["TimesFmModelForPrediction"],
    ...         "num_hidden_layers": 1,
    ...         "hidden_size": 16,
    ...         "intermediate_size": 16,
    ...         "head_dim": 8,
    ...         "num_attention_heads": 4,
    ...         "context_length": 8,
    ...         "horizon_length": 6,
    ...         "patch_length": 2,
    ...         "quantiles": [0.25, 0.5, 0.75],
    ...     },
    ...     validation_split=0.1,
    ...     training_args={
    ...         "max_steps": 1,
    ...         "eval_steps": 1,
    ...     },
    ... )
    >>> forecaster.pretrain(y_panel)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pretrain": True,
        "authors": ["rajatsen91", "siriuz42", "geetu040"],
        # rajatsen91, siriuz42 for google-research/timesfm
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]>=4.52.0"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="google/timesfm-2.5-200m-transformers",
        config=None,
        device_map="cpu",
        dtype=None,
        quantization_config=None,
        forward_kwargs=None,
        peft_config=None,
        validation_split=0.2,
        training_args=None,
        compute_loss_func=None,
        compute_metrics=None,
        callbacks=None,
    ):
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.forward_kwargs = forward_kwargs
        self.peft_config = peft_config
        self.validation_split = validation_split
        self.training_args = training_args
        self.compute_loss_func = compute_loss_func
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks

        super().__init__()

    def _pretrain(self, y, X=None, fh=None):
        """Pretrain forecaster on panel/global data (first batch).

        private _pretrain containing the core logic, called from pretrain

        Writes to self:
            Sets pretrained model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex (guaranteed Panel or Hierarchical)
            Panel or hierarchical time series data to pretrain on.
            The last index level is time, all other levels identify instances.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series.
        fh : ForecastingHorizon or None, optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        self.model_ = self._load_model()

        context_length = self.model_.config.context_length
        horizon_length = self.model_.config.horizon_length

        from transformers import Trainer, TrainingArguments

        if self.validation_split is not None:
            y_train, y_eval = temporal_train_test_split(
                y, test_size=self.validation_split
            )
        else:
            y_train = y
            y_eval = None

        train = PyTorchDataset(
            series_list=_prepare_series_list(y_train),
            context_length=context_length,
            horizon_length=horizon_length,
        )

        eval = None
        if (
            self.validation_split is not None
            and len(y_eval) >= context_length + horizon_length
        ):
            eval = PyTorchDataset(
                series_list=_prepare_series_list(y_eval),
                context_length=context_length,
                horizon_length=horizon_length,
            )
        elif self.validation_split is not None:
            warn(
                "Skipping TimesFM evaluation dataset creation: validation split "
                "length is smaller than context_length + horizon_length "
                f"(observed={len(y_eval)}, required>="
                f"{context_length + horizon_length}). Training continues "
                "without evaluation.",
                stacklevel=2,
            )

        training_args = (
            deepcopy(self.training_args) if self.training_args is not None else {}
        )
        training_args = TrainingArguments(**training_args)

        trainer = Trainer(
            model=self.model_,
            args=training_args,
            train_dataset=train,
            eval_dataset=eval,
            compute_loss_func=self.compute_loss_func,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        trainer.train()

        self.model_ = trainer.model

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.

            * if self.get_tag("capability:multivariate")==False:
              guaranteed to be univariate (e.g., single-column for DataFrame)
            * if self.get_tag("capability:multivariate")==True: no restrictions apply,
              the method should handle uni- and multivariate y appropriately

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self.model_ = self._load_model()
        self.context_ = y

    def _validate_predict_fh(self, fh):
        """Return relative forecasting horizon indices after capacity validation."""
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1

        horizon_length = self.model_.config.horizon_length
        if np.max(preds_idx) >= horizon_length:
            raise ValueError(
                "Requested forecasting horizon exceeds TimesFM model capacity: "
                f"max requested step={np.max(preds_idx) + 1}, "
                f"configured horizon_length={horizon_length}. "
                "Use a shorter horizon or a model/config with a larger "
                "horizon_length."
            )

        return fh, preds_idx

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        import torch

        self.model_ = self._load_model()

        fh, preds_idx = self._validate_predict_fh(fh)

        past_values = self.context_
        past_values = np.expand_dims(past_values, axis=0)
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        forward_kwargs = {} if not self.forward_kwargs else self.forward_kwargs
        output = self.model_(past_values=past_values, **forward_kwargs)

        preds = output.mean_predictions
        preds = preds.ravel()
        preds = preds[preds_idx]
        preds = preds.detach().float().cpu().numpy()
        preds = pd.Series(
            preds,
            index=fh.to_absolute(self._cutoff)._values,
            name=self.context_.name,
        )

        return preds

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        import torch

        self.model_ = self._load_model()

        quantiles = self.model_.config.quantiles
        past_values = self.context_

        fh, preds_idx = self._validate_predict_fh(fh)

        if alpha is None:
            alpha = quantiles
        alpha = [round(i, 3) for i in alpha]
        quantiles = [round(i, 3) for i in quantiles]
        if not set(alpha).issubset(set(quantiles)):
            raise ValueError(
                "Requested quantiles are not all available in model config: "
                f"requested={alpha}, available={quantiles}."
            )
        quantiles_idx = np.array([quantiles.index(i) for i in alpha])

        past_values = np.expand_dims(past_values, axis=0)
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        forward_kwargs = {} if not self.forward_kwargs else self.forward_kwargs
        output = self.model_(past_values=past_values, **forward_kwargs)

        preds = output.full_predictions
        preds = preds.squeeze(0)
        preds = preds[preds_idx]
        preds = preds[:, quantiles_idx]
        preds = preds.detach().float().cpu().numpy()

        index = fh.to_absolute(self._cutoff)._values
        name = self.context_.name if self.context_.name is not None else 0
        columns = pd.MultiIndex.from_product([[name], alpha])
        pred_quantiles = pd.DataFrame(
            data=preds,
            index=index,
            columns=columns,
        )

        return pred_quantiles

    def _load_model(self):
        """Load or retrieve a cached TimesFM model instance.

        Returns
        -------
        model : transformers.PreTrainedModel
            Loaded model according to ``self.device_map`` and
            ``self.dtype``. If ``self.model_`` already exists, it is
            returned directly.
        """
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        self.model_ = _CachedTimesFM2(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device_map=self.device_map,
            dtype=self.dtype,
            quantization_config=self.quantization_config,
            peft_config=self.peft_config,
        ).load()

        return self.model_

    def _get_unique_key(self):
        """Build cache key for the multiton model loader.

        Returns
        -------
        key : str
            Deterministic string representation of model-loading attributes used
            by :class:`_CachedTimesFM2`.
        """
        key = {
            "model_path": self.model_path,
            "config": self.config,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "quantization_config": self.quantization_config,
            "peft_config": self.peft_config,
        }
        return str(sorted(key.items()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        intervals = [0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
        quantiles = sorted(q for p in intervals for q in ((1 - p) / 2, (1 + p) / 2))
        quantiles = [0.1] + quantiles
        return [
            {
                "model_path": None,
                "config": {
                    "architectures": ["TimesFmModelForPrediction"],
                    "num_hidden_layers": 1,
                    "hidden_size": 16,
                    "intermediate_size": 16,
                    "head_dim": 8,
                    "num_attention_heads": 4,
                    "context_length": 8,
                    "horizon_length": 6,
                    "patch_length": 2,
                    "quantiles": quantiles,
                },
                "validation_split": 0.1,
                "training_args": {
                    "max_steps": 1,
                    "eval_steps": 1,
                },
                "device_map": "cpu",
            },
            {
                "model_path": None,
                "config": {
                    "architectures": ["TimesFm2_5ModelForPrediction"],
                    "num_hidden_layers": 1,
                    "hidden_size": 8,
                    "intermediate_size": 4,
                    "head_dim": 2,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "context_length": 8,
                    "horizon_length": 6,
                    "patch_length": 2,
                    "quantiles": quantiles,
                },
                "validation_split": 0.1,
                "training_args": {
                    "max_steps": 1,
                    "eval_steps": 1,
                },
            },
        ]


@_multiton
class _CachedTimesFM2:
    """Multiton-backed cache wrapper for a loaded TimesFM model.

    Instances are keyed externally (via :class:`TimesFM2Forecaster`) so that
    repeated forecasters with equivalent loading parameters can share a single
    model instance in memory.

    Parameters
    ----------
    key : str
        Multiton key (stored for traceability).
    model_path : str or None
        Model identifier/path for ``from_pretrained``. If ``None``, create model
        from config only.
    config : transformers.PretrainedConfig or dict or None
        Configuration used for model loading/creation.
    peft_config : peft.PeftConfig or None
        Optional PEFT wrapping configuration.
    device_map : str, dict, int, or torch.device
        Device placement for loading models.
    dtype : torch.dtype or str or None
        Data type used for model loading.
    quantization_config : transformers.quantizers.HfQuantizer or None
        Quantization configuration used for pretrained model loading.
    """

    def __init__(
        self,
        key,
        model_path,
        config,
        peft_config,
        device_map,
        dtype,
        quantization_config,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.peft_config = peft_config
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.model_ = None

    def load(self):
        """Load model if needed and return cached instance.

        Returns
        -------
        model : transformers.PreTrainedModel
            Loaded TimesFM prediction model according to ``self.device_map``,
            ``self.dtype``, and ``self.quantization_config``.

        Notes
        -----
        - If ``model_path`` is set, loads weights via ``from_pretrained`` with
          ``device_map``, ``dtype``, and optional ``quantization_config``.
        - If ``model_path`` is ``None``, initializes from config.
        - If ``peft_config`` is provided with a pretrained ``model_path``,
          wraps the model with PEFT after base loading and device placement.
        - ``quantization_config`` and ``peft_config`` are not applied in the
          config-only initialization path.
        """
        if self.model_ is not None:
            return self.model_

        if self.model_path is not None:
            self.model_ = self._load_from_path()
        else:
            self.model_ = self._load_randomly()

        return self.model_

    def _load_from_path(self):
        """Load pretrained TimesFM weights from ``self.model_path``.

        This path applies ``device_map``, ``dtype``, ``quantization_config``,
        and optional PEFT wrapping.
        """
        from transformers import AutoConfig

        config = self.config
        if not config:
            config = AutoConfig.from_pretrained(self.model_path)
        if isinstance(config, dict):
            config_class = _get_timesfm_config_class(config)
            config = config_class.from_dict(config)

        model_class = _get_timesfm_model_class(config)

        model = model_class.from_pretrained(
            self.model_path,
            config=config,
            device_map=self.device_map,
            dtype=self.dtype,
            quantization_config=self.quantization_config,
        )

        if self.peft_config is not None and _check_soft_dependencies(
            "peft", severity="error"
        ):
            from peft import get_peft_model

            model = get_peft_model(model, deepcopy(self.peft_config))

        return model

    def _load_randomly(self):
        """Initialize a TimesFM model from config without pretrained weights.

        This path applies ``device_map`` and ``dtype`` after model construction.
        It does not apply ``quantization_config`` or ``peft_config``.
        """
        from transformers import TimesFmConfig

        config = self.config
        if not config:
            config = TimesFmConfig()
        if isinstance(config, dict):
            config_class = _get_timesfm_config_class(config)
            config = config_class.from_dict(config)

        model_class = _get_timesfm_model_class(config)
        model = model_class(config)
        model = model.to(self.device_map)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)

        return model


def _get_timesfm_model_class(config):
    """Resolve TimesFM prediction model class from a config object.

    Parameters
    ----------
    config : transformers.PretrainedConfig
        Config object expected to contain ``architectures``.

    Returns
    -------
    model_class : type
        ``transformers`` model class referenced by ``config.architectures[0]``.

    Notes
    -----
    Defaults to ``TimesFmModelForPrediction`` when ``architectures`` is unset.
    For TimesFM-2.5 architecture, a soft dependency warning is emitted for
    ``transformers>=5.3.0``.
    """
    import transformers

    architectures = getattr(config, "architectures", None) or [
        "TimesFmModelForPrediction"
    ]
    if architectures[0] == "TimesFm2_5ModelForPrediction":
        _check_soft_dependencies("transformers>=5.3.0", severity="warning")
    return getattr(transformers, architectures[0])


def _get_timesfm_config_class(config):
    """Resolve TimesFM config class from a config dictionary.

    Parameters
    ----------
    config : dict
        Config dictionary with optional ``architectures`` entry.

    Returns
    -------
    config_class : type
        Matching ``transformers`` config class.

    Notes
    -----
    Maps ``*ModelForPrediction`` architecture names to ``*Config``.
    For TimesFM-2.5 architecture, a soft dependency warning is emitted for
    ``transformers>=5.3.0``.
    """
    import transformers

    architectures = config.get("architectures") or ["TimesFmModelForPrediction"]
    if architectures[0] == "TimesFm2_5ModelForPrediction":
        _check_soft_dependencies("transformers>=5.3.0", severity="warning")
    config_class_name = architectures[0].replace("ModelForPrediction", "Config")
    return getattr(transformers, config_class_name)


def _prepare_series_list(data):
    """Convert panel/hierarchical DataFrame into list of 1D numpy series.

    Parameters
    ----------
    data : pd.DataFrame
        MultiIndex time series table where the last index level is time.

    Returns
    -------
    series_list : list of np.ndarray
        Flattened list of univariate instance-column series, each as a
        ``numpy`` array.
    """
    instance_levels = list(range(data.index.nlevels - 1))
    groupby_level = instance_levels[0] if len(instance_levels) == 1 else instance_levels

    series_list = []
    for _, group in data.groupby(level=groupby_level):
        for col in group.columns:
            series_list.append(group[col].to_numpy())

    return series_list


def _pad_series(series, seq_len):
    """Left-pad a series with zeros to at least a target length.

    Parameters
    ----------
    series : np.ndarray
        Input 1D numeric series.
    seq_len : int
        Required minimum sequence length.

    Returns
    -------
    padded : np.ndarray
        ``series`` left-padded with zeros if ``len(series) < seq_len``;
        otherwise unchanged.
    """
    pad_length = seq_len - len(series)
    if pad_length >= 0:
        series = np.pad(
            series,
            (pad_length, 0),
            mode="constant",
            constant_values=0,
        )
    return series


class PyTorchDataset:
    """Sliding-window dataset for TimesFM pretraining with ``transformers.Trainer``.

    Each sample is a contiguous window of length
    ``context_length + horizon_length`` split into:

    - ``past_values`` (context part)
    - ``future_values`` (target horizon part)

    Parameters
    ----------
    series_list : list of np.ndarray
        List of 1D training series.
    context_length : int
        Number of historical points in each sample.
    horizon_length : int
        Number of forecast points in each sample.

    Raises
    ------
    ValueError
        If no valid training samples can be generated. This happens when all
        input series have length ``<= horizon_length``.
    """

    def __init__(self, series_list, context_length, horizon_length):
        self.series_list = series_list
        self.context_length = context_length
        self.horizon_length = horizon_length

        min_length = context_length + horizon_length
        self.samples = []
        for series in series_list:
            if len(series) <= horizon_length:
                continue
            series = _pad_series(series, min_length)
            for start in range(len(series) - min_length + 1):
                self.samples.append(series[start : start + min_length])

        if not self.samples:
            raise ValueError(
                "No training samples were generated for TimesFM pretraining. "
                "Provide at least one series with length greater than "
                f"horizon_length (horizon_length={horizon_length})."
            )

    def __len__(self):
        """Return number of generated samples.

        Returns
        -------
        int
            Dataset size.
        """
        return len(self.samples)

    def __getitem__(self, i):
        """Return one training sample as tensors.

        Parameters
        ----------
        i : int
            Sample index.

        Returns
        -------
        sample : dict
            Dictionary with keys ``past_values`` and ``future_values``, both
            ``torch.float32`` tensors.
        """
        import torch

        sample = self.samples[i]

        past_values = sample[: self.context_length]
        future_values = sample[self.context_length :]

        past_values = torch.tensor(past_values, dtype=torch.float32)
        future_values = torch.tensor(future_values, dtype=torch.float32)

        return {
            "past_values": past_values,
            "future_values": future_values,
        }
