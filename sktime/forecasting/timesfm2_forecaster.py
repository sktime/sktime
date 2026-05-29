# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface for the HuggingFace TimesFM-2.x forecasting model series."""

__author__ = ["rajatsen91", "siriuz42", "geetu040"]
# rajatsen91 for google-research/timesfm

__all__ = ["TimesFM2Forecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TimesFM2Forecaster(BaseForecaster):
    """TimesFM-2.x forecaster via Hugging Face transformers.

    TimesFM is a pretrained time-series foundation model developed by
    Google Research for zero-shot forecasting. This forecaster wraps the
    Hugging Face ``transformers`` implementation of TimesFM-2.x and exposes
    it through the sktime forecasting interface.

    Parameters
    ----------
    model_path : str, default="google/timesfm-2.5-200m-transformers"
        Hugging Face model identifier or local path to a TimesFM-2.x checkpoint.
        If ``None``, a model is initialized from ``config`` or from the default
        ``transformers.TimesFmConfig``.
    config : transformers.TimesFmConfig or dict, optional (default=None)
        Configuration passed to the Hugging Face model loader. If ``model_path``
        is not ``None``, this overrides or supplies the model configuration in
        ``from_pretrained``. If ``model_path`` is ``None``, it is used to
        initialize the model from configuration.
    forward_kwargs : dict, optional (default=None)
        Keyword arguments passed directly to the Hugging Face model forward
        method during inference. See the Hugging Face TimesFM forward
        documentation for supported model-specific options such as
        ``forecast_context_len``, ``truncate_negative``, or
        ``force_flip_invariance``.
        See [5]_ for TimesFM-2.0 and [6]_ for TimesFM-2.5.
    validation_split : float, default=0.2
        Fraction of data reserved for validation. This parameter is retained for
        compatibility with Hugging Face training-style interfaces; the current
        zero-shot implementation does not fine-tune the model.
    training_args : transformers.TrainingArguments or dict, optional (default=None)
        Training arguments placeholder for future fine-tuning support. Not used
        by the current zero-shot implementation.
    compute_loss_func : callable, optional (default=None)
        Custom loss function placeholder for future fine-tuning support. Not
        used by the current zero-shot implementation.
    compute_metrics : callable or list of callable, optional (default=None)
        Metric function or functions placeholder for future fine-tuning support.
        Not used by the current zero-shot implementation.
    callbacks : list, optional (default=None)
        Hugging Face Trainer callbacks placeholder for future fine-tuning
        support. Not used by the current zero-shot implementation.
    device : str, default="cpu"
        Device on which to run the model, for example ``"cpu"``, ``"cuda"``,
        or ``"cuda:0"``.

    References
    ----------
    .. [1] Das, A., Kong, W., Sen, R., and Zhou, Y. (2024).
       A decoder-only foundation model for time-series forecasting. CoRR.
       https://arxiv.org/abs/2310.10688
    .. [2] https://github.com/google-research/timesfm
    .. [3] https://huggingface.co/google/timesfm-2.5-200m-transformers
    .. [4] https://huggingface.co/google/timesfm-2.0-500m-pytorch
    .. [5] https://huggingface.co/docs/transformers/en/model_doc/timesfm#transformers.TimesFmModelForPrediction.forward
    .. [6] https://huggingface.co/docs/transformers/en/model_doc/timesfm2_5#transformers.TimesFm2_5ModelForPrediction.forward

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm2_forecaster import TimesFM2Forecaster
    >>> y = load_airline()
    >>> forecaster = TimesFM2Forecaster(
    ...     model_path="google/timesfm-2.5-200m-transformers",
    ...     device="cpu",
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pretrain": False,
        "authors": ["rajatsen91", "siriuz42", "geetu040"],
        # rajatsen91, siriuz42 for google-research/timesfm
        "maintainers": ["geetu040"],
        "python_dependencies": ["torch", "transformers"],
    }

    def __init__(
        self,
        model_path="google/timesfm-2.5-200m-transformers",
        config=None,
        forward_kwargs=None,
        validation_split=0.2,
        training_args=None,
        compute_loss_func=None,
        compute_metrics=None,
        callbacks=None,
        device="cpu",
    ):
        self.model_path = model_path
        self.config = config
        self.forward_kwargs = forward_kwargs
        self.validation_split = validation_split
        self.training_args = training_args
        self.compute_loss_func = compute_loss_func
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.device = device

        super().__init__()

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

        past_values = self.context_
        past_values = np.expand_dims(past_values, axis=0)
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1

        forward_kwargs = {} if self.forward_kwargs is None else self.forward_kwargs
        output = self.model_(past_values=past_values, **forward_kwargs)

        preds = output.mean_predictions
        preds = preds.ravel()
        preds = preds[preds_idx]
        preds = preds.detach().cpu().numpy()
        preds = pd.Series(
            preds,
            index=fh.to_absolute(self._cutoff)._values,
            name=self.context_.name,
        )

        return preds

    def __getstate__(self):
        """Return state for pickling, excluding the model object."""
        state = self.__dict__.copy()
        if "model_" in state:
            state["model_"] = None
        return state

    def __setstate__(self, state):
        """Restore state; the model will be reloaded on next prediction."""
        self.__dict__.update(state)

    def _load_model(self):
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        self.model_ = _CachedTimesFM2(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            device=self.device,
        ).load()

        return self.model_

    def _get_unique_key(self):
        return str(
            sorted(
                {
                    "model_path": self.model_path,
                    "config": self.config,
                    "device": self.device,
                }.items()
            )
        )

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
        return [
            {
                "model_path": None,
                "config": {
                    "architectures": ["TimesFm2ModelForPrediction"],
                    "num_hidden_layers": 1,
                    "hidden_size": 16,
                    "intermediate_size": 16,
                    "head_dim": 8,
                    "num_attention_heads": 4,
                },
                "device": "cpu",
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
                },
            },
        ]


@_multiton
class _CachedTimesFM2:
    """Cached TimesFM 2.5 model instance."""

    def __init__(self, key, model_path, config, device):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.device = device
        self.model_ = None

    def load(self):
        if self.model_ is not None:
            return self.model_

        from transformers import AutoModelForTimeSeriesPrediction, TimesFmConfig

        if self.model_path is not None:
            self.model_ = AutoModelForTimeSeriesPrediction.from_pretrained(
                self.model_path,
                config=self.config,
            )
            self.model_ = self.model_.to(self.device)
            return self.model_

        config = self.config
        if config is None:
            config = TimesFmConfig()
        if isinstance(config, dict):
            config = TimesFmConfig.from_dict(config)

        self.model_ = AutoModelForTimeSeriesPrediction.from_config(config)
        self.model_ = self.model_.to(self.device)
        return self.model_
