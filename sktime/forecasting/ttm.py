# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TinyTimeMixer for forecasting."""

__author__ = ["ajati", "wgifford", "vijaye12", "geetu040"]
# ajati, wgifford, vijaye12 for ibm-granite code

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from skbase.utils.stdout_mute import StdoutMute

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.dependencies import _safe_import
from sktime.utils.warnings import warn

torch = _safe_import("torch")
Dataset = _safe_import("torch.utils.data.Dataset")


class TinyTimeMixerForecaster(_BaseGlobalForecaster):
    """
    TinyTimeMixer Forecaster for Zero-Shot Forecasting of Multivariate Time Series.

    Wrapping implementation in [1]_ of method proposed in [2]_. See [3]_
    for tutorial by creators.

    TinyTimeMixer (TTM) are compact pre-trained models for Time-Series Forecasting,
    open-sourced by IBM Research. With less than 1 Million parameters, TTM introduces
    the notion of the first-ever "tiny" pre-trained models for Time-Series Forecasting.

    **Fit Strategies: Full, Minimal, and Zero-shot**

    This model supports three fit strategies: *zero-shot* for direct predictions without
    training, *minimal* fine-tuning for lightweight adaptation to new data, and *full*
    fine-tuning for comprehensive model training. The selected strategy is determined
    by the model's fit_strategy parameter

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
       ``fit`` exceeds the default ``config.prediction_length``,
       the configuration is updated to reflect ``max(fh)``.

    6. **Model Architecture**: The final configuration is used to construct the
       *model architecture*.

    7. **Pretrained Weights**: *pretrained weights* are loaded from the ``model_path``,
       these weights are then aligned and loaded into the *model architecture*.

    8. **Weight Alignment**: However sometimes, *pretrained weights* do not align with
       the *model architecture*, because the config was changed which created a
       *model architecture* of different size than the default one.
       This causes some of the weights in *model architecture* to be reinitialized
       randomly instead of using the pre-trained weights.

    **Training Strategies**:

    - **Zero-shot Forecasting**: When all the *pre-trained weights* are correctly
      aligned with the *model architecture*, fine-tuing part is bypassed and
      the model preforms zero-short forecasting.

    - **Minimal Fine-tuning**: When not all the *pre-trained weights* are correctly
      aligned with the *model architecture*, rather some weights are re-initialized,
      these re-initialized weights are fine-tuned on the provided data.

    - **Full Fine-tuning**:  The model is *fully fine-tuned* on new data, updating *all
      parameters*. This approach offers maximum adaptation to the dataset but requires
      more computational resources.

    **Exogenous Variables Support**

    TTM supports exogenous variables (external factors) that can improve
    forecasting accuracy. The model accepts exogenous variables that are
    known for both the historical period and the future forecasting horizon.

    When using exogenous variables:
    - The X parameter should contain exogenous data covering both
      past and future periods
    - Exogenous variables must have the same index structure as the target series
    - For prediction, exogenous data must extend into the forecasting horizon

    Parameters
    ----------
    model_path : str, default="ibm/TTM"
        Path to the Huggingface model to use for forecasting.
        This can be either:

        - The name of a Huggingface repository (e.g., "ibm/TTM")

        - A local path to a folder containing model files in a format supported
          by transformers. In this case, ensure that the directory contains all
          necessary files (e.g., configuration, tokenizer, and model weights).

        - If this parameter is *None*, fit_strategy should be *full* to allow
          full fine tuning of the model loaded from pretrained/provided config,
          else ValueError is raised.

    revision: str, default="main"
        Revision of the model to use:

        - "main": For loading model with context_length of 512
          and prediction_length of 96.

        - "1024_96_v1": For loading model with context_length of 1024
          and prediction_length of 96.

        This param becomes irrelevant when model_path is None

    validation_split : float, default=0.2
        Fraction of the data to use for validation

    config : dict, default={}
        Configuration to use for the model. See the ``transformers``
        documentation for details.

    training_args : dict, default={}
        Training arguments to use for the model. See ``transformers.TrainingArguments``
        for details.
        Note that the ``output_dir`` argument is required.

    compute_metrics : list, default=None
        List of metrics to compute during training. See ``transformers.Trainer``
        for details.

    callbacks : list, default=[]
        List of callbacks to use during training. See ``transformers.Trainer``

    broadcasting : bool, default=False
        if True, multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from ``predict``.

    use_source_package : bool, default=False
        If True, the model and configuration will be loaded directly from the source
        package ``tsfm_public.models.tinytimemixer``. This is useful if you
        want to bypass the local version of the package or when working in an
        environment where the latest updates from the source package are needed.
        If False, the model and configuration will be loaded from the local
        version of package maintained in sktime because of model's unavailability
        on pypi.
        To install the source package, follow the instructions here [4]_.

    fit_strategy : str, default="minimal"
        Strategy to use for fitting (fine-tuning) the model. This can be one of
        the following:
        - "zero-shot": Uses pre-trained model as it is. If model path is *None*
          with this strategy, ValueError is raised.
        - "minimal": Fine-tunes only a small subset of the model parameters,
          allowing for quick adaptation with limited computational resources.
          If model path is *None* with this strategy, ValueError is raised.
        - "full": Fine-tunes all model parameters, which may result in better
          performance but requires more computational power and time. Allows
          model path to be *None*.

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer
    .. [2] Ekambaram, V., Jati, A., Dayama, P., Mukherjee, S.,
           Nguyen, N.H., Gifford, W.M., Reddy, C. and Kalagnanam, J., 2024.
           Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced
           Zero/Few-Shot Forecasting of Multivariate Time Series. CoRR.
    .. [3] https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/tutorial/ttm_tutorial.ipynb
    .. [4] https://github.com/ibm-granite/granite-tsfm/tree/ttm

    Examples
    --------
    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TinyTimeMixerForecaster()
    >>> # performs zero-shot forecasting, as default config (unchanged) is used
    >>> forecaster.fit(y, fh=[1, 2, 3])
    TinyTimeMixerForecaster(...)
    >>> y_pred = forecaster.predict()

    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.datasets import load_tecator
    >>>
    >>> # load multi-index dataset
    >>> y = load_tecator(
    ...     return_type="pd-multiindex",
    ...     return_X_y=False
    ... )
    >>> y.drop(['class_val'], axis=1, inplace=True)
    >>>
    >>> # global forecasting on multi-index dataset
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path=None,
    ...     fit_strategy="full",
    ...     config={
    ...             "context_length": 8,
    ...             "prediction_length": 2
    ...     },
    ...     training_args={
    ...         "num_train_epochs": 1,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 32,
    ...     },
    ... )
    >>>
    >>> # model initialized with random weights due to None model_path
    >>> # and trained with the full strategy.
    >>> forecaster.fit(y, fh=[1, 2, 3])
    TinyTimeMixerForecaster(...)
    >>> y_pred = forecaster.predict()

    Example with exogenous variables:

    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.datasets import load_longley
    >>> from sktime.split import temporal_train_test_split
    >>> y, X = load_longley()
    >>> y_train, _, X_train, X_future = temporal_train_test_split(y, X, test_size=2)
    >>>
    >>> # Initialize forecaster
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path=None,
    ...     fit_strategy="full",
    ...     config={
    ...         "context_length": 8,
    ...         "prediction_length": 2
    ...     },
    ...     training_args={
    ...         "max_steps": 10,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 4,
    ...         "report_to": "none",
    ...     },
    ... )
    >>>
    >>> # Fit with exogenous variables
    >>> forecaster.fit(y_train, X=X_train, fh=[1, 2])
    TinyTimeMixerForecaster(...)
    >>>
    >>> # Predict with exogenous variables
    >>> y_pred = forecaster.predict(X=X_future)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ajati", "wgifford", "vijaye12", "geetu040"],
        # ajati, wgifford, vijaye12 for ibm-granite code
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers", "torch", "accelerate>=0.26.0"],
        # estimator type
        # --------------
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
        "scitype:y": "both",
        "capability:exogenous": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:libs": ["sktime.libs.granite_ttm"],
    }

    def __init__(
        self,
        model_path="ibm/TTM",
        revision="main",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
        broadcasting=False,
        use_source_package=False,
        fit_strategy="minimal",
    ):
        super().__init__()
        self.model_path = model_path
        self.revision = revision
        self.config = config
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
        self.validation_split = validation_split
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.broadcasting = broadcasting
        self.use_source_package = use_source_package
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
        from transformers import Trainer, TrainingArguments

        if self.use_source_package:
            from tsfm_public.models.tinytimemixer import (
                TinyTimeMixerConfig,
                TinyTimeMixerForPrediction,
            )
        elif _check_soft_dependencies("torch", severity="error"):
            from sktime.libs.granite_ttm import (
                TinyTimeMixerConfig,
                TinyTimeMixerForPrediction,
            )

        # Initialize to self.config, then adjust accordingly
        config = self.config
        if self.model_path is None:
            if self.fit_strategy != "full":
                raise ValueError(
                    "Invalid configuration: 'model_path' is set to None."
                    "This requires 'fit_strategy' to be 'full'."
                    "Please set 'fit_strategy' to 'full' or provide a valid model path."
                )
            # Load tinytimemixer config
            config = TinyTimeMixerConfig()
            # call to initialize attributes like num_patchess
            config.check_and_init_preprocessing()
        else:
            # Get the pretrained model Configuration
            config = TinyTimeMixerConfig.from_pretrained(
                self.model_path,
                revision=self.revision,
            )

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self._config)

        # validate patches in configuration
        # context_length / num_patches == patch_length == patch_stride
        # if this condition is not satisfied in the configuration
        # this error is raised in forward pass of the model
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (384x4 and 32x64)
        context_length = _config.get("context_length")
        num_patches = _config.get("num_patches")
        patch_length = _config.get("patch_length")
        patch_stride = _config.get("patch_stride")
        patch_size = context_length / num_patches
        if patch_size != patch_length or patch_size != patch_stride:
            # update the config here
            patch_size = max(1, int(patch_size))
            _config["patch_length"] = patch_size
            _config["patch_stride"] = patch_size
            _config["num_patches"] = _config["context_length"] // patch_size

            msg = (
                "Invalid configuration detected. "
                "The provided values do not satisfy the required condition:\n"
                "context_length / num_patches == patch_length == patch_stride\n"
                "Provided configuration:\n"
                f"- context_length: {context_length}\n"
                f"- num_patches: {num_patches}\n"
                f"- patch_length: {patch_length}\n"
                f"- patch_stride: {patch_stride}\n"
                "Configuration has been automatically updated to:\n"
                f"- context_length: {context_length}\n"
                f"- num_patches: {_config['num_patches']}\n"
                f"- patch_length: {_config['patch_length']}\n"
                f"- patch_stride: {_config['patch_stride']}"
            )
            warn(msg)

        if fh is not None:
            _config["prediction_length"] = max(
                *(fh.to_relative(self._cutoff)._values),
                _config["prediction_length"],
            )

        config = config.from_dict(_config)

        if self.model_path is not None:
            # Load the the pretrained model with updated config
            self.model, info = TinyTimeMixerForPrediction.from_pretrained(
                self.model_path,
                revision=self.revision,
                config=config,
                output_loading_info=True,
                ignore_mismatched_sizes=True,
            )
        else:
            # Initialize model with default config
            self.model = TinyTimeMixerForPrediction(config=config)

        if self.fit_strategy == "zero-shot":
            if len(info["mismatched_keys"]) > 0:
                raise ValueError(
                    "Fit strategy is 'zero-shot', but the model weights in the"
                    "configuration are mismatched compared to the pretrained model."
                    "Please ensure they match."
                )
            return
        elif self.fit_strategy == "minimal":
            if len(info["mismatched_keys"]) == 0:
                return  # No need to fit
            # Freeze all loaded parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Adjust requires_grad property of model weights based on info
            for key in info["mismatched_keys"]:
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
            # Handle exogenous variables split if provided
            X_train, X_eval = None, None
            if X is not None:
                X_train, X_eval = temporal_train_test_split(
                    X, test_size=self.validation_split
                )
        else:
            y_train = y
            y_eval = None
            X_train = X
            X_eval = None

        train = PyTorchDataset(
            y=y_train,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
            X=X_train,
        )

        eval = None
        if self.validation_split is not None:
            eval = PyTorchDataset(
                y=y_eval,
                context_length=config.context_length,
                prediction_length=config.prediction_length,
                X=X_eval,
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

        with StdoutMute() as _:
            trainer.train()

        # Get the model
        self.model = trainer.model

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
        past_values, observed_mask = _pad_truncate(
            hist, self.model.config.context_length
        )

        past_values = (
            torch.tensor(past_values).to(self.model.dtype).to(self.model.device)
        )
        observed_mask = (
            torch.tensor(observed_mask).to(self.model.dtype).to(self.model.device)
        )

        # Handle exogenous variables if provided
        future_values = None
        if X is not None:
            # Process exogenous variables for prediction
            if isinstance(X.index, pd.MultiIndex):
                exog_data = _frame2numpy(X)
            else:
                exog_data = np.expand_dims(X.values, axis=0)

            # Extract future exogenous values for the prediction horizon
            # X should contain future values that cover the prediction horizon
            prediction_length = self.model.config.prediction_length
            context_length = self.model.config.context_length

            # Get the last context_length + prediction_length values from exog_data
            if exog_data.shape[1] >= context_length + prediction_length:
                # Take the last prediction_length values as future exogenous
                future_exog = exog_data[:, -prediction_length:, :]
                future_values = (
                    torch.tensor(future_exog).to(self.model.dtype).to(self.model.device)
                )

        self.model.eval()
        outputs = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
            future_values=future_values,
        )
        pred = outputs.prediction_outputs.detach().cpu().numpy()

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
            # batch_size * num_timestams, n_cols
            pred.reshape(-1, pred.shape[-1]),
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
                "training_args": {
                    "max_steps": 5,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 4,
                    "report_to": "none",
                },
            },
            {
                "model_path": "ibm/TTM",
                "revision": "main",
                "config": {
                    "context_length": 3,
                    "prediction_length": 2,
                },
                "validation_split": 0.2,
                "training_args": {
                    "max_steps": 5,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 4,
                    "report_to": "none",
                },
            },
        ]
        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in test_params]
        test_params.extend(params_broadcasting)
        return test_params


def _pad_truncate(data, seq_len, pad_value=0):
    """
    Pad or truncate a numpy array.

    Parameters
    ----------
    - data: numpy array of shape (batch_size, original_seq_len, n_dims)
    - seq_len: sequence length to pad or truncate to
    - pad_value: value to use for padding

    Returns
    -------
    - padded_data: array padded or truncated to (batch_size, seq_len, n_dims)
    - mask: mask indicating padded elements (1 for existing; 0 for missing)
    """
    batch_size, original_seq_len, n_dims = data.shape

    # Truncate or pad each sequence in data
    if original_seq_len > seq_len:
        truncated_data = data[:, -seq_len:, :]
        mask = np.ones_like(truncated_data)
    else:
        truncated_data = np.pad(
            data,
            ((0, 0), (seq_len - original_seq_len, 0), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        mask = np.zeros_like(truncated_data)
        mask[:, -original_seq_len:, :] = 1

    return truncated_data, mask


def _same_index(data):
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(lambda x: x.equals(data.iloc[0])).all(), (
        "All series must has the same index"
    )
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data):
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, context_length, prediction_length, X=None):
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
        X : ndarray, optional (default=None)
            Exogenous variables, shape (n_sequences, n_timestamps, n_exog_dims)
            Should cover both past and future values for the entire time range
        """
        self.context_length = context_length
        self.prediction_length = prediction_length

        # multi-index conversion for y
        if isinstance(y.index, pd.MultiIndex):
            self.y = _frame2numpy(y)
        else:
            self.y = np.expand_dims(y.values, axis=0)

        # Handle exogenous variables
        self.X = None
        if X is not None:
            if isinstance(X.index, pd.MultiIndex):
                self.X = _frame2numpy(X)
            else:
                self.X = np.expand_dims(X.values, axis=0)

        self.n_sequences, self.n_timestamps, _ = self.y.shape
        self.single_length = max(
            1, (self.n_timestamps - self.context_length - self.prediction_length + 1)
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

        # Handle edge case where data is shorter than context + prediction length
        if self.n_timestamps < self.context_length + self.prediction_length:
            # Use all available data for past values and pad if necessary
            available_context = min(
                self.context_length, self.n_timestamps - self.prediction_length
            )
            available_context = max(1, available_context)

            past_values = self.y[n, :available_context, :]
            # Pad if context is shorter than required
            if past_values.shape[0] < self.context_length:
                padding_needed = self.context_length - past_values.shape[0]
                past_values = np.pad(
                    past_values,
                    ((padding_needed, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            # For future values, take the last available data points
            future_start = max(0, self.n_timestamps - self.prediction_length)
            future_values = self.y[
                n, future_start : future_start + self.prediction_length, :
            ]

            # Pad future values if needed
            if future_values.shape[0] < self.prediction_length:
                padding_needed = self.prediction_length - future_values.shape[0]
                future_values = np.pad(
                    future_values,
                    ((0, padding_needed), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            past_values = self.y[n, m : m + self.context_length, :]
            future_values = self.y[
                n,
                m + self.context_length : m
                + self.context_length
                + self.prediction_length,
                :,
            ]

        observed_mask = np.ones_like(past_values)

        result = {
            "past_values": tensor(past_values).float(),
            "observed_mask": tensor(observed_mask).float(),
            "future_values": tensor(future_values).float(),
        }

        # Add exogenous variables if available
        if self.X is not None:
            if self.n_timestamps < self.context_length + self.prediction_length:
                # Handle short sequences for exogenous variables
                available_context = min(
                    self.context_length, self.n_timestamps - self.prediction_length
                )
                available_context = max(1, available_context)

                past_exog = self.X[n, :available_context, :]
                if past_exog.shape[0] < self.context_length:
                    padding_needed = self.context_length - past_exog.shape[0]
                    past_exog = np.pad(
                        past_exog,
                        ((padding_needed, 0), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )

                future_start = max(0, self.n_timestamps - self.prediction_length)
                future_exog = self.X[
                    n, future_start : future_start + self.prediction_length, :
                ]

                if future_exog.shape[0] < self.prediction_length:
                    padding_needed = self.prediction_length - future_exog.shape[0]
                    future_exog = np.pad(
                        future_exog,
                        ((0, padding_needed), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
            else:
                # Normal case
                past_exog = self.X[n, m : m + self.context_length, :]
                future_exog = self.X[
                    n,
                    m + self.context_length : m
                    + self.context_length
                    + self.prediction_length,
                    :,
                ]

            # Concatenate past and future exogenous for the full sequence
            full_exog = np.concatenate([past_exog, future_exog], axis=0)
            result["exogenous_values"] = tensor(full_exog).float()

        return result
