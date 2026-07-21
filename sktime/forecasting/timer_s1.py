# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Timer-S1 forecaster for ``sktime``.

This module provides an ``sktime`` forecaster wrapping the local Timer-S1
``transformers`` model implementation. It supports:

- zero-shot prediction through :meth:`fit` + :meth:`predict`
- quantile prediction through :meth:`predict_quantiles`

Model training and fine-tuning are not supported at the moment, though they may
be added in future. Calling :meth:`fit` only loads the model and stores the
observed series as forecasting context.
"""

__author__ = ["WenWeiTHU", "geetu040"]
# WenWeiTHU for bytedance-research/Timer-S1

__all__ = ["TimerS1Forecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TimerS1Forecaster(BaseForecaster):
    """Timer-S1 forecaster via Hugging Face ``transformers``.

    This forecaster wraps Timer-S1 prediction models [1]_, [2]_ from Hugging
    Face and exposes them through the ``sktime`` forecasting interface.

    The primary workflow is ``fit`` for zero-shot inference setup, which loads
    the model and stores history. It does not train or fine-tune model weights.
    Passing ``model_path=None`` initializes a Timer-S1 model from ``config``
    instead of loading pretrained weights. These random weights cannot be
    trained through this estimator at the moment and are mainly useful for tests
    or local experimentation.

    Notes
    -----
    - Timer-S1 training is currently not supported, but may be added in future.
      The estimator performs only zero-shot forecasting from a loaded or
      randomly initialized model.
    - Quantile prediction is only available for quantiles present in
      ``model.config.quantiles``.
    - Loaded models are cached via a multiton helper keyed by model-loading
      inputs to avoid repeated model instantiation.
    - The default Timer-S1 checkpoint has 8 billion parameters. For most
      hardware, reduced-memory loading with ``dtype`` and ``quantization_config``,
      or a pre-quantized checkpoint, is recommended.

    Parameters
    ----------
    model_path : str, default="bytedance-research/Timer-S1"
        Hugging Face repository identifier or local path to a Timer-S1 checkpoint.
        If ``None``, a model is created from ``config``.
    config : TimerS1Config or dict, optional (default=None)
        Model configuration used when ``model_path=None``. If provided as a
        ``dict``, it is converted with ``TimerS1Config.from_dict``. If ``None``
        and ``model_path=None``, the default ``TimerS1Config`` is used. This
        path creates random weights; the estimator does not currently provide
        training for those weights.
    device_map : str, dict, int, or torch.device, default="cpu"
        Device placement following the ``transformers`` ``device_map`` naming
        convention, for example ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or
        ``"auto"``.
    dtype : torch.dtype or str, optional (default=None)
        Data type used for model loading, following the ``transformers``
        ``dtype`` convention, for example ``torch.float16``,
        ``torch.bfloat16``, or ``"auto"``.
    quantization_config : transformers.quantizers.HfQuantizer, optional
        Valid quantization configuration object compatible with
        ``transformers.PreTrainedModel.from_pretrained`` [3]_.
    forward_kwargs : dict, optional (default=None)
        Additional keyword arguments forwarded to ``model.generate(...)`` during
        :meth:`predict` and :meth:`predict_quantiles`.
    deterministic : bool, default=False
        Whether point predictions should reset the ``transformers`` random seed
        before generation. Currently this is applied in predict methods.

    References
    ----------
    .. [1] Liu, Y., Su, X., Wang, S., Zhang, H., Liu, H., Wang, Y.,
       Ye, Z., Xiang, Y., Wang, J., and Long, M. (2026).
       Timer-S1: A Billion-Scale Time Series Foundation Model with Serial
       Scaling. arXiv. https://arxiv.org/abs/2603.04791
    .. [2] Timer-S1 model card:
       https://huggingface.co/bytedance-research/Timer-S1
    .. [3] Quantization docs:
       https://huggingface.co/docs/transformers/en/main_classes/quantization

    Examples
    --------
    Simple zero-shot forecasting with the default Timer-S1 checkpoint:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> y = load_airline()
    >>> # By default, loads bytedance-research/Timer-S1.
    >>> forecaster = TimerS1Forecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Reduced-memory inference for the 8-billion-parameter model:

    >>> import torch  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> from transformers import BitsAndBytesConfig  # doctest: +SKIP
    >>> y = load_airline()
    >>> forecaster = TimerS1Forecaster(  # doctest: +SKIP
    ...     model_path="bytedance-research/Timer-S1",
    ...     forward_kwargs={"revin": True},
    ...     device_map="auto",
    ...     dtype=torch.bfloat16,
    ...     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Loading a quantized smaller model directly:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> y = load_airline()
    >>> forecaster = TimerS1Forecaster(  # doctest: +SKIP
    ...     model_path="geetu040/Timer-S1-quantized-4bit",
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Randomly initialized local model, useful for tests or local experimentation.
    This model is not trained by ``fit``; the weights stay random and should not
    be used as a trained forecaster:

    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> forecaster = TimerS1Forecaster(  # doctest: +SKIP
    ...     model_path=None,
    ...     config={
    ...         "hidden_size": 16,
    ...         "intermediate_size": 16,
    ...         "num_attention_heads": 4,
    ...         "num_experts": 4,
    ...         "num_hidden_layers": 1,
    ...         "num_mtp_tokens": 1,
    ...     },
    ...     deterministic=True,
    ... )

    Quantile prediction:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timer_s1 import TimerS1Forecaster
    >>> y = load_airline()
    >>> forecaster = TimerS1Forecaster(  # doctest: +SKIP
    ...     model_path="geetu040/Timer-S1-quantized-4bit",
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 2, 3],
    ...     alpha=[0.1, 0.5, 0.9],
    ... )
    """

    _tags = {
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "authors": ["WenWeiTHU", "geetu040"],
        # WenWeiTHU for bytedance-research/Timer-S1
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]>4.57.0,<5.0.0"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="bytedance-research/Timer-S1",
        config=None,
        device_map="cpu",
        dtype=None,
        quantization_config=None,
        forward_kwargs=None,
        deterministic=False,
    ):
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.forward_kwargs = forward_kwargs
        self.deterministic = deterministic

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Load the model and store history for zero-shot forecasting.

        private _fit containing the core logic, called from fit

        This method does not train or fine-tune Timer-S1 weights. Training may
        be added in future.

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
        import transformers

        if self.deterministic:
            transformers.set_seed(42)

        self.model_ = self._load_model()

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        horizon_length = np.max(preds_idx) + 1

        quantiles = np.array(self.model_.config.quantiles)
        weights = (quantiles[:-1] + quantiles[1:]) / 2
        weights = np.concatenate([[0.0], weights, [1.0]])
        weights = np.diff(weights)

        past_values = self.context_
        past_values = np.expand_dims(past_values, axis=0)
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        forward_kwargs = {} if not self.forward_kwargs else self.forward_kwargs
        output = self.model_.generate(
            past_values, max_new_tokens=horizon_length, **forward_kwargs
        )

        preds = output.squeeze(axis=0)
        preds = preds.detach().float().cpu().numpy()
        preds = np.average(preds, weights=weights, axis=0)
        preds = preds[preds_idx]
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
        import transformers

        if self.deterministic:
            transformers.set_seed(42)

        self.model_ = self._load_model()

        quantiles = self.model_.config.quantiles
        past_values = self.context_

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        horizon_length = np.max(preds_idx) + 1

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
        output = self.model_.generate(
            past_values, max_new_tokens=horizon_length, **forward_kwargs
        )

        preds = output.squeeze(axis=0)
        preds = preds.T
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
        """Load or retrieve a cached Timer-S1 model instance.

        Returns
        -------
        model : transformers.PreTrainedModel
            Loaded model according to ``self.device_map`` and
            ``self.dtype``. If ``self.model_`` already exists, it is
            returned directly.
        """
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        self.model_ = _CachedTimerS1(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device_map=self.device_map,
            dtype=self.dtype,
            quantization_config=self.quantization_config,
        ).load()

        return self.model_

    def _get_unique_key(self):
        """Build cache key for the multiton model loader.

        Returns
        -------
        key : str
            Deterministic string representation of model-loading attributes used
            by :class:`_CachedTimerS1`.
        """
        key = {
            "model_path": self.model_path,
            "config": self.config,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "quantization_config": self.quantization_config,
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

        test_params = []

        test_param_1 = {
            "model_path": None,
            "config": {
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_attention_heads": 4,
                "num_experts": 4,
                "num_hidden_layers": 1,
                "num_mtp_tokens": 1,
                "quantiles": quantiles,
                "use_cache": False,
            },
            "deterministic": True,
        }
        test_params.append(test_param_1)

        test_param_2 = test_param_1.copy()
        test_param_2.update(
            {
                "forward_kwargs": {"revin": True},
            }
        )
        test_params.append(test_param_2)

        return test_params


@_multiton
class _CachedTimerS1:
    """Multiton-backed cache wrapper for a loaded Timer-S1 model.

    Instances are keyed externally (via :class:`TimerS1Forecaster`) so that
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
    device_map : str, dict, int, or torch.device
        Device placement for loading models.
    dtype : torch.dtype or str or None
        Data type used for model loading.
    quantization_config : transformers.quantizers.HfQuantizer or None
        Quantization configuration used for model loading.
    """

    def __init__(
        self,
        key,
        model_path,
        config,
        device_map,
        dtype,
        quantization_config,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.model_ = None

    def load(self):
        """Load model if needed and return cached instance.

        Returns
        -------
        model : transformers.PreTrainedModel
            Loaded Timer-S1 prediction model according to ``self.device_map``,
            ``self.dtype``, and ``self.quantization_config``.

        Notes
        -----
        - If ``model_path`` is set, loads weights via ``from_pretrained``.
        - If ``model_path`` is ``None``, initializes from config with random
          weights. These weights are not trained by the estimator.
        """
        if self.model_ is not None:
            return self.model_

        if self.model_path is not None:
            self.model_ = self._load_from_path()
        else:
            self.model_ = self._load_randomly()

        return self.model_

    def _load_from_path(self):
        """Load Timer-S1 model weights from ``self.model_path``."""
        from sktime.libs.timer_s1 import TimerS1ForPrediction

        model = TimerS1ForPrediction.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            dtype=self.dtype,
            quantization_config=self.quantization_config,
            trust_remote_code=True,
        )

        return model

    def _load_randomly(self):
        """Initialize a Timer-S1 model randomly from config."""
        from sktime.libs.timer_s1 import TimerS1Config, TimerS1ForPrediction

        warn(
            "Initializing Timer-S1 from config creates random weights. "
            "Timer-S1 training is not supported by this estimator at the "
            "moment, so these weights will stay random and are only suitable "
            "for tests or local experimentation.",
            UserWarning,
            stacklevel=2,
        )

        config = deepcopy(self.config)
        if not config:
            config = TimerS1Config()
        if isinstance(config, dict):
            config = TimerS1Config.from_dict(config)

        model = TimerS1ForPrediction(config)
        model = model.to(self.device_map)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)

        return model
