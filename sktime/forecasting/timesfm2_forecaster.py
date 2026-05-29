# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface for the HuggingFace TimesFM-2.x forecasting model series."""

__author__ = ["rajatsen91", "siriuz42", "geetu040"]
# rajatsen91 for google-research/timesfm

__all__ = ["TimesFM2Forecaster"]

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.split import temporal_train_test_split
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
        Fraction of data reserved for validation when ``training_args`` is
        supplied. If ``None``, no validation dataset is passed to the Hugging
        Face Trainer.
    training_args : transformers.TrainingArguments or dict, optional (default=None)
        Training arguments for Hugging Face Trainer.
        Custom loss function passed to Hugging Face Trainer during fine-tuning.
    compute_metrics : callable or list of callable, optional (default=None)
        Metric function or functions passed to Hugging Face Trainer during
        fine-tuning.
    callbacks : list, optional (default=None)
        Hugging Face Trainer callbacks used during fine-tuning.
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
    .. [7] https://huggingface.co/docs/transformers/v5.9.0/en/main_classes/trainer#transformers.TrainingArguments

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
        "capability:pred_int": True,
        "capability:pretrain": True,
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

        train = _TimesFM2WindowDataset(
            series_list=_prepare_series_list(y_train),
            context_length=context_length,
            horizon_length=horizon_length,
        )

        eval = None
        if (
            self.validation_split is not None
            and len(y_eval) >= context_length + horizon_length
        ):
            eval = _TimesFM2WindowDataset(
                series_list=_prepare_series_list(y_eval),
                context_length=context_length,
                horizon_length=horizon_length,
            )
        elif self.validation_split is not None:
            warn(
                "Could not create a TimesFM validation dataset because the "
                "validation split is shorter than context_length + horizon_length "
                f"({len(y_eval)} < {context_length + horizon_length}). "
                "Training will continue without evaluation.",
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

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1

        horizon_length = self.model_.config.horizon_length
        if np.max(preds_idx) >= horizon_length:
            raise ValueError(
                "The requested forecasting horizon extends beyond the TimesFM "
                f"model horizon_length. The maximum requested relative horizon "
                f"is {np.max(preds_idx) + 1}, but model.config.horizon_length is "
                f"{horizon_length}."
            )

        past_values = self.context_
        past_values = np.expand_dims(past_values, axis=0)
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

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

        horizon_length = self.model_.config.horizon_length
        quantiles = self.model_.config.quantiles
        past_values = self.context_

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)
        preds_idx = fh._values.values - 1
        if np.max(preds_idx) >= horizon_length:
            raise ValueError(
                "The requested forecasting horizon extends beyond the TimesFM "
                f"model horizon_length. The maximum requested relative horizon "
                f"is {np.max(preds_idx) + 1}, but model.config.horizon_length is "
                f"{horizon_length}."
            )

        if alpha is None:
            alpha = quantiles
        alpha = [round(i, 3) for i in alpha]
        quantiles = [round(i, 3) for i in quantiles]
        if not set(alpha).issubset(set(quantiles)):
            raise ValueError(
                "The requested quantiles are different than the ones in config "
                f"alpha: {alpha} and valid quantiles in config: {quantiles}"
            )
        quantiles_idx = np.array([quantiles.index(i) for i in alpha])

        past_values = np.expand_dims(past_values, axis=0)
        past_values = torch.from_numpy(past_values)
        past_values = past_values.to(self.model_.dtype)
        past_values = past_values.to(self.model_.device)

        forward_kwargs = {} if self.forward_kwargs is None else self.forward_kwargs
        output = self.model_(past_values=past_values, **forward_kwargs)

        preds = output.full_predictions
        preds = preds.squeeze(0)
        preds = preds[preds_idx]
        preds = preds[:, quantiles_idx]
        preds = preds.detach().cpu().numpy()

        index = fh.to_absolute(self._cutoff)._values
        name = self.context_.name if self.context_.name is not None else 0
        columns = pd.MultiIndex.from_product([[name], alpha])
        pred_quantiles = pd.DataFrame(
            data=preds,
            index=index,
            columns=columns,
        )

        return pred_quantiles

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
        intervals = [0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
        quantiles = sorted(q for p in intervals for q in ((1 - p) / 2, (1 + p) / 2))
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


def _prepare_series_list(data):
    instance_levels = list(range(data.index.nlevels - 1))
    groupby_level = instance_levels[0] if len(instance_levels) == 1 else instance_levels

    series_list = []
    for _, group in data.groupby(level=groupby_level):
        for col in group.columns:
            series_list.append(group[col].to_numpy())

    return series_list


def _pad_series(series, seq_len):
    pad_length = seq_len - len(series)
    if pad_length >= 0:
        series = np.pad(
            series,
            (pad_length, 0),
            mode="constant",
            constant_values=0,
        )
    return series


class _TimesFM2WindowDataset:
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
                "No training samples could be generated. "
                f"Each series must have length greater than horizon_length "
                f"({horizon_length})."
            )

    def __len__(self):
        """Return length of dataset."""
        return len(self.samples)

    def __getitem__(self, i):
        """Return data point."""
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
