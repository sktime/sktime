# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of Timer for forecasting."""

__author__ = ["WenWeiTHU", "ZDandsomSP", "Sohaib-Ahmed21"]
# WenWeiTHU, ZDandsomSP for thuml code

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.forecasting.ttm import _frame2numpy, _pad_truncate
from sktime.split import temporal_train_test_split

if _check_soft_dependencies("torch", severity="none"):
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class TimerForecaster(_BaseGlobalForecaster):
    """
    Timer Forecaster for Zero-Shot Forecasting of Univariate Time Series.

    Wrapping implementation in [1]_ of method proposed in [2]_. See [3]_
    for hugging face tutorial.

    Timer: Generative Pre-trained Transformers Are Large Time Series Models
    It introduces a groundbreaking approach to leveraging transformers for
    accurate and scalable time series forecasting.

    **Fit Strategies: Full, Minimal, and Zero-shot**

    This model supports three fit strategies: *zero-shot* for direct predictions without
    training, *minimal* fine-tuning for lightweight adaptation to new data, and *full*
    fine-tuning for comprehensive model training. The selected strategy is determined
    by the model's fit_strategy parameter.

    **Initialization Process**:

    1. **Model Path**: The ``model_path`` parameter points to a local folder or
       huggingface repo that contains both *configuration files*
       and *pretrained weights*.

    2. **Default Configuration**: The model loads its default configuration from the
       *configuration files*.

    3. **Custom Configuration**: Users can provide a custom configuration via the
       ``config`` parameter during model initialization.

    4. **Configuration Override**: If custom configuration is provided,
       it overrides the default configuration.

    5. **Forecasting Horizon**: If the forecasting horizon (``fh``) specified during
       ``fit`` exceeds the default ``config.input_token_len``,
       the configuration is updated to reflect ``max(fh)``.

    6. **Model Architecture**: The final configuration is used to construct the
       *model architecture*.

    7. **Pretrained Weights**: *pretrained weights* are loded from the ``model_path``,
       these weights are then aligned and loaded into the *model architechture*.

    8. **Weight Alignment**: However sometimes, *pretrained weights* do not align with
       the *model architechture*, because the config was changed which created a
       *model architechture* of different size than the default one.
       This causes some of the weights in *model architechture* to be reinitialized
       randomly instead of using the pre-trained weights.

    **Forecasting Modes**:

    - **Zero-shot Forecasting**: When all the *pre-trained weights* are correctly
      aligned with the *model architechture*, fine-tuing part is bypassed and
      the model preforms zero-short forecasting.

    - **Minimal Fine-tuning**: When not all the *pre-trained weights* are correctly
      aligned with the *model architechture*, rather some weights are re-initialized,
      these re-initialized weights are fine-tuned on the provided data.

    - **Full Fine-tuning**:  The model is *fully fine-tuned* on new data, updating *all
      parameters*. This approach offers maximum adaptation to the dataset but requires
      more computational resources.

    Parameters
    ----------
    model_path : str, required parameter
        Path to the Huggingface model to use for forecasting.
        This can be either:

        - The name of a Huggingface repository (e.g., "thuml/timer-base-84m")

        - A local path to a folder containing model files in a format supported
          by transformers. In this case, ensure that the directory contains all
          necessary files (e.g., configuration, tokenizer, and model weights).

    trust_remote_code : bool, default=False
        Whether or not to allow for custom models defined on the Hub in their own
        modeling files. This option should only be set to True for repositories you
        trust and in which you have read the code, as it will execute code present on
        the Hub on your local machine.

    validation_split : float, default=0.2
        Fraction of the data to use for validation

    config : dict, default=None
        Configuration to use for the model. See the ``transformers``
        documentation for details.

    training_args : dict, default=None
        Training arguments to use for the model. See ``transformers.TrainingArguments``
        for details.
        Note that the ``output_dir`` argument is required.

    compute_metrics : list, default=None
        List of metrics to compute during training. See ``transformers.Trainer``
        for details.

    callbacks : list, default=None
        List of callbacks to use during training. See ``transformers.Trainer``

    broadcasting : bool, default=False
        if True, multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from ``predict``.

    fit_strategy : str, default="minimal"
        Strategy to use for fitting (fine-tuning) the model. This can be one of
        the following:
        - "zero-shot": Uses pre-trained model as it is.
        - "minimal": Fine-tunes only a small subset of the model parameters,
          allowing for quick adaptation with limited computational resources.
        - "full": Fine-tunes all model parameters, which may result in better
          performance but requires more computational power and time.


    References
    ----------
    .. [1] https://github.com/thuml/Large-Time-Series-Model/tree/main
    .. [2] Liu, Y., Zhang, H., Li, C., Huang, X., Wang, J. and Long, M., 2024.
           Timer: Transformers for Time Series Analysis at Scale. CoRR.
    .. [3] https://huggingface.co/thuml/timer-base-84m/tree/main

    Examples
    --------
    >>> from sktime.forecasting.timer import TimerForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TimerForecaster(
    ...     model_path="thuml/timer-base-84m",
    ...     trust_remote_code=True,
    ...     training_args={
    ...         "output_dir": "test_output",
    ...     },
    ... )  # doctest: +SKIP
    >>> # performs zero-shot forecasting, as default config (unchanged) is used
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP

    >>> from sktime.forecasting.timer import TimerForecaster
    >>> from sktime.datasets import load_tecator
    >>>
    >>> # load multi-index dataset
    >>> y = load_tecator(
    ...     return_type="pd-multiindex",
    ...     return_X_y=False
    ... )
    >>> y.drop(['class_val'], axis=1, inplace=True)
    >>>
    >>> # global forecasting on multi-index dataset with custom config
    >>> forecaster = TimerForecaster(
    ...     model_path="thuml/timer-base-84m",
    ...     trust_remote_code=True,
    ...     config={
    ...         "num_hidden_layers": 8,
    ...         "input_token_len": 32,
    ...         "output_token_lens": [24]
    ...     },
    ...     training_args={
    ...         "num_train_epochs": 1,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 32,
    ...     },
    ... )  # doctest: +SKIP
    >>>
    >>> # model is fine-tuned, as a config different from default is provided.
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    """

    _tags = {
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "authors": ["WenWeiTHU", "ZDandsomSP", "Sohaib-Ahmed21"],
        # WenWeiTHU, ZDandsomSP for thuml code
        "maintainers": ["Sohaib-Ahmed21"],
        "python_dependencies": ["transformers", "torch"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self,
        model_path: str,
        trust_remote_code=False,
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
        broadcasting=False,
        fit_strategy="minimal",
    ):
        super().__init__()
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.config = config
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
        self.validation_split = validation_split
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.broadcasting = broadcasting
        self.fit_strategy = fit_strategy

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.DataFrame",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            Trainer,
            TrainingArguments,
        )

        # Load model and extract config
        config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self._config)

        if fh is not None:
            _config["output_token_lens"][0] = max(
                *(fh.to_relative(self._cutoff)._values + 1),
                _config["output_token_lens"][0],
            )

        config = config.from_dict(_config)

        prediction_length = config.output_token_lens[0]
        context_length = config.input_token_len

        # Load model with the updated config
        self.model, info = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
            trust_remote_code=self.trust_remote_code,
        )

        if self.fit_strategy == "zero-shot":
            if len(info["mismatched_keys"]) > 0 or len(info["missing_keys"]) > 0:
                raise ValueError(
                    "Fit strategy is 'zero-shot', but the model weights in the"
                    "configuration are mismatched or missing compared to the pre-"
                    "trained model.Please ensure they match. See pretrained config at"
                    "https://huggingface.co/thuml/timer-base-84m/blob/main/config.json"
                )
            return

        elif self.fit_strategy == "minimal":
            if len(info["mismatched_keys"]) == 0 and len(info["missing_keys"]) == 0:
                return  # No need to fit

            # Freeze all loaded parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Adjust requires_grad for layers with mismatched sizes
            for key, _, _ in info["mismatched_keys"]:
                _model = self.model
                for attr_name in key.split(".")[:-1]:
                    _model = getattr(_model, attr_name)
                _model.weight.requires_grad = True

            # Adjust requires_grad for layers with missing keys
            for key in info["missing_keys"]:
                _model = self.model
                for attr_name in key.split(".")[:-1]:
                    _model = getattr(_model, attr_name)
                _model.weight.requires_grad = True

        elif self.fit_strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Unknown fit strategy")

        if self.validation_split is not None:
            y_train, y_eval = temporal_train_test_split(
                y, test_size=self.validation_split
            )
        else:
            y_train = y
            y_eval = None

        train = PyTorchDataset(
            y=y_train,
            context_length=context_length,
            prediction_length=prediction_length,
        )

        eval = None
        if self.validation_split is not None:
            eval = PyTorchDataset(
                y=y_eval,
                context_length=context_length,
                prediction_length=prediction_length,
            )

        # Get Training Configuration
        training_args = TrainingArguments(**self._training_args)
        # Get the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train,
            eval_dataset=eval,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        # Train the model
        trainer.train()

        # Get the model
        self.model = trainer.model

        return self

    def _predict(self, fh, X, y=None):
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

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        _y = y if self._global_forecasting else self._y

        # multi-index conversion goes here
        if isinstance(_y.index, pd.MultiIndex):
            hist = _frame2numpy(_y)
        else:
            hist = np.expand_dims(_y.values, axis=0)

        # hist.shape: (batch_size, n_timestamps, n_cols)

        # truncate or pad to match sequence length
        past_values, _ = _pad_truncate(hist, self.model.config.input_token_len)

        # remove last dim as it is always 1 in univariate
        past_values = past_values.squeeze(-1)
        # past_values.shape: (batch_size, n_timestamps)

        past_values = (
            torch.tensor(past_values).to(self.model.dtype).to(self.model.device)
        )

        self.model.eval()
        pred = self.model.generate(
            inputs=past_values,
            max_new_tokens=max(fh._values),
        )

        # converting pred datatype
        if isinstance(_y.index, pd.MultiIndex):
            ins = np.array(
                list(np.unique(_y.index.droplevel(-1)).repeat(pred.shape[1]))
            )
            ins = [ins[..., i] for i in range(ins.shape[-1])] if ins.ndim > 1 else [ins]
            idx = (
                ForecastingHorizon(range(1, pred.shape[1] + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values.tolist()
                * pred.shape[0]
            )
            index = pd.MultiIndex.from_arrays(
                ins + [idx],
                names=_y.index.names,
            )
        else:
            index = (
                ForecastingHorizon(range(1, pred.shape[1] + 1))
                .to_absolute(self._cutoff)
                ._values
            )

        pred = pd.DataFrame(
            # batch_size * num_timestamps, n_cols where n_cols=1 in univariate
            pred.reshape(-1, 1),
            index=index,
            columns=_y.columns,
        )

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = pred.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        pred = pred.loc[dateindex]
        pred.index.names = _y.index.names

        return pred

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
        test_params = [
            {
                "model_path": "thuml/timer-base-84m",
                "trust_remote_code": True,
                "training_args": {
                    "max_steps": 4,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 4,
                    "report_to": "none",
                },
            },
            {
                "model_path": "thuml/timer-base-84m",
                "trust_remote_code": True,
                "config": {
                    "input_token_len": 16,
                    "output_token_lens": 8,
                },
                "validation_split": 0.2,
                "training_args": {
                    "max_steps": 5,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 4,
                    "report_to": "none",
                },
            },
            {
                "model_path": "thuml/timer-base-84m",
                "trust_remote_code": True,
                "config": {
                    "input_token_len": 20,
                    "output_token_lens": 12,
                },
                "validation_split": 0.2,
                "training_args": {
                    "max_steps": 5,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 4,
                    "report_to": "none",
                },
                "fit_strategy": "full",
            },
        ]
        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in test_params]
        test_params.extend(params_broadcasting)
        return test_params


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, context_length, prediction_length):
        """
        Initialize the dataset.

        Parameters
        ----------
        y : ndarray
            The time series data, shape (n_sequences, n_timestamps, n_dims)
        context_length : int
            The length of the past values
        prediction_length : int
            The length of the future values
        """
        self.context_length = context_length
        self.prediction_length = prediction_length

        # multi-index conversion
        if isinstance(y.index, pd.MultiIndex):
            self.y = _frame2numpy(y)
        else:
            self.y = np.expand_dims(y.values, axis=0)

        self.n_sequences, self.n_timestamps, _ = self.y.shape
        self.single_length = (
            self.n_timestamps - self.context_length - self.prediction_length + 1
        )

    def __len__(self):
        """Return the length of the dataset."""
        # Calculate the number of samples that can be created from each sequence
        return self.single_length * self.n_sequences

    def __getitem__(self, i):
        """Return data point."""
        from torch import tensor

        m = i % self.single_length
        n = i // self.single_length

        past_values = self.y[n, m : m + self.context_length, :]
        future_values = self.y[
            n,
            m + self.context_length : m + self.context_length + self.prediction_length,
            :,
        ]
        past_values = past_values.reshape(-1)
        future_values = future_values.reshape(-1)

        return {
            "input_ids": tensor(past_values).float(),
            "labels": tensor(future_values).float(),
        }
