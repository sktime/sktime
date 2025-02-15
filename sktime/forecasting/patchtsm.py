# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TinyTimeMixer for forecasting."""

__author__ = [
    "ajati",
    "vijaye12",
    "gsinthong",
    "namctin",
    "wmgifford",
    "kashif",
    "AffanBinFaisal",
]
# ajati, vijaye12, gsinthong, namctin, wmgifford, kashif

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.warnings import warn

if _check_soft_dependencies("torch", severity="none"):
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class PatchTSMixerForecaster(_BaseGlobalForecaster):
    """
    TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting.

    HuggingFace implementation in [1]_ of method proposed in [2]_

    PatchTSMixer is a deep learning architecture designed for time series
    forecasting, leveraging patch-based tokenization and mixer layers
    to capture temporal dependencies efficiently. It partitions time series
    data into patches, enabling better feature extraction and long-term
    trend modeling. The model is pre-trained for zero/few-shot forecasting,
    making it effective for multivariate datasets with limited supervision.
    Its design enhances both accuracy and computational efficiency in
    forecasting tasks.

    **Zero-shot and Fine-tuning**:

    This model is designed as a versatile foundation model, capable
    of both zero-shot forecasting and fine-tuning on custom datasets.
    The choice between zero-shot and fine-tuning is managed internally
    and is determined by the final configuration used to initialize the
    model.

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

    - **Fine-tuning**: When not all the *pre-trained weights* are correctly aligned
      with the *model architechture*, rather some weights are re-initialized,
      these re-initialized weights are fine-tuned on the provided data.

    Parameters
    ----------
    model_path : str, default="ibm-granite/granite-timeseries-patchtsmixer"
        Path to the Huggingface model to use for forecasting.
        This can be either:

        - The name of a Huggingface repository (e.g.,
         "ibm-granite/granite-timeseries-patchtsmixer")

        - A local path to a folder containing model files in a format supported
          by transformers. In this case, ensure that the directory contains all
          necessary files (e.g., configuration, tokenizer, and model weights).

    revision: str, default="main"
        Revision of the model to use:

        - "main": For loading model with context_length of 512
          and prediction_length of 96.

        - "1024_96_v1": For loading model with context_length of 1024
          and prediction_length of 96.

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

    References
    ----------
    .. [1] https://github.com/huggingface/transformers/blob/main/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py
    .. [2] Vijay Ekambaram, Arindam Jati, Nam Nguyen,
           Phanwadee Sinthong, Jayant Kalgnanam, 2024.
           TSMixer: Lightweight MLP-Mixer Model for
           Multivariate Time Series Forecasting. CoRR.

    Examples
    --------
    >>> from sktime.forecasting.patchtsm import PatchTSMixerForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = PatchTSMixerForecaster()
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP

    >>> from sktime.forecasting.patchtsm import PatchTSMixerForecaster
    >>> from sktime.datasets import load_tecator
    >>> # load multi-index dataset
    >>> y = load_tecator(
    ...         return_type="pd-multiindex",
    ...         return_X_y=False
    ...     )
    >>> y.drop(['class_val'], axis=1, inplace=True)
    >>> # global forecasting on multi-index dataset
    >>> forecaster = PatchTSMixerForecaster(
    ...     config={
    ...             "context_length": 8,
    ...             "prediction_length": 2
    ...         },
    ...         training_args={
    ...             "num_train_epochs": 1,
    ...             "output_dir": "test_output",
    ...             "per_device_train_batch_size": 32,
    ...         },
    ... )  # doctest: +SKIP

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
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "authors": [
            "ajati",
            "vijaye12",
            "gsinthong",
            "namctin",
            "wmgifford",
            "kashif",
            "AffanBinFaisal",
        ],
        # ajati, vijaye12, gsinthong, namctin, wmgifford, kashif
        "maintainers": ["AffanBinFaisal"],
        "python_dependencies": ["transformers", "torch"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self,
        model_path="ibm-granite/granite-timeseries-patchtsmixer",
        revision="main",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
        broadcasting=False,
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

        Private _fit method containing the core logic, called from fit.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            The time series to which the forecaster is fitted.
            Guaranteed to be of an mtype in self.get_tag("y_inner_mtype").
            If self.get_tag("scitype:y")=="univariate":
                Guaranteed to have a single column/variable.
            If self.get_tag("scitype:y")=="multivariate":
                Guaranteed to have 2 or more columns.
            If self.get_tag("scitype:y")=="both": no restrictions apply.
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon specifying the steps ahead to predict.
            Required here if self.get_tag("requires-fh-in-fit")==True.
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict.
        X : sktime time series object, optional (default=None)
            Exogenous time series for forecasting.
            Guaranteed to be of an mtype in self.get_tag("X_inner_mtype").

        Returns
        -------
        self : reference to self
        """
        from transformers import (
            PatchTSMixerConfig,
            PatchTSMixerForPrediction,
            Trainer,
            TrainingArguments,
        )

        # Load a pre-trained PatchTSMixer configuration from the specified model path
        config = PatchTSMixerConfig.from_pretrained(
            self.model_path,  # Path to the pre-trained model
            revision=self.revision,  # Model version or revision
        )

        # Convert the loaded configuration to a dictionary
        _config = config.to_dict()

        # Update the configuration with additional user-defined parameters
        _config.update(self._config)

        # Extract key parameters from the updated configuration
        context_length = _config.get(
            "context_length"
        )  # Total length of the input context window
        num_patches = _config.get(
            "num_patches"
        )  # Number of patches to divide the context into
        patch_length = _config.get("patch_length")  # Length of each patch
        patch_stride = _config.get("patch_stride")  # Stride between patches

        # Compute patch size based on context length and number of patches
        patch_size = (
            context_length / num_patches
        )  # Ensure num_patches is non-zero to avoid division error

        # Validate and adjust patch size configuration if necessary
        if patch_size != patch_length or patch_size != patch_stride:
            # Ensure patch size is a valid integer
            patch_size = max(1, int(patch_size))

            # Update config with corrected values
            _config["patch_length"] = patch_size
            _config["patch_stride"] = patch_size
            _config["num_patches"] = _config["context_length"] // patch_size

            # Warn about automatic configuration adjustment
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

        # Update prediction length based on the forecasting horizon
        if fh is not None:
            _config["prediction_length"] = max(
                *(
                    fh.to_relative(self._cutoff)._values
                ),  # Convert FH to relative values
                _config["prediction_length"],
            )

        # Apply the updated configuration
        config = config.from_dict(_config)

        # Load the PatchTSMixer model with the updated configuration
        self.model, info = PatchTSMixerForPrediction.from_pretrained(
            self.model_path,
            revision=self.revision,
            config=config,
            output_loading_info=True,  # Get details about loading mismatches
            ignore_mismatched_sizes=True,  # Ignore size mismatches in parameters
        )

        # If no mismatched keys were found, return early (no need to retrain)
        if len(info["mismatched_keys"]) == 0:
            return

        # Freeze all model parameters to prevent unnecessary updates
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze weights for layers that had mismatched keys
        for key, _, _ in info["mismatched_keys"]:
            _model = self.model
            for attr_name in key.split(".")[:-1]:  # Traverse nested attributes
                _model = getattr(_model, attr_name)
            _model.weight.requires_grad = True  # Enable gradient updates

        # Perform train-validation split if validation data is provided
        if self.validation_split is not None:
            y_train, y_eval = temporal_train_test_split(
                y, test_size=self.validation_split
            )
        else:
            y_train = y
            y_eval = None

        # Create PyTorch datasets for training and validation
        train = PyTorchDataset(
            y=y_train,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
        )

        eval = None
        if self.validation_split is not None:
            eval = PyTorchDataset(
                y=y_eval,
                context_length=config.context_length,
                prediction_length=config.prediction_length,
            )

        # Initialize training arguments with user-defined settings
        training_args = TrainingArguments(**self._training_args)

        # Initialize the Trainer with the model, datasets, and training configuration
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train,
            eval_dataset=eval,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        # Train the model using the provided training arguments
        trainer.train()

        # Save the trained model
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
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
            If not passed in _fit, guaranteed to be passed here.
        X : sktime time series object, optional (default=None)
            Exogeneous time series for the forecast.

        Returns
        -------
        y_pred : sktime time series object
            Point predictions with the same type as "y_inner_mtype" tag.
        """
        import torch

        # If forecasting horizon is not provided, use the stored one
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        # Use global forecasting data if applicable, otherwise use fitted y
        _y = y if self._global_forecasting else self._y

        # Convert multi-index data to numpy array if applicable
        if isinstance(_y.index, pd.MultiIndex):
            hist = _frame2numpy(_y)
        else:
            hist = np.expand_dims(_y.values, axis=0)  # Reshape for batch processing

        # Ensure the historical data matches the required sequence length
        past_values, observed_mask = _pad_truncate(
            hist, self.model.config.context_length
        )

        # Convert past values and mask to PyTorch tensors and move to the correct device
        past_values = (
            torch.tensor(past_values).to(self.model.dtype).to(self.model.device)
        )
        observed_mask = (
            torch.tensor(observed_mask).to(self.model.dtype).to(self.model.device)
        )

        # Set the model to evaluation mode
        self.model.eval()

        # Perform inference using the trained model
        outputs = self.model(past_values=past_values, observed_mask=observed_mask)
        pred = outputs.prediction_outputs.detach().cpu().numpy()

        # Convert predictions to a pandas DataFrame with appropriate indexing
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
            index = pd.MultiIndex.from_arrays(ins + [idx], names=_y.index.names)
        else:
            index = (
                ForecastingHorizon(range(1, pred.shape[1] + 1))
                .to_absolute(self._cutoff)
                ._values
            )

        pred = pd.DataFrame(
            pred.reshape(-1, pred.shape[-1]), index=index, columns=_y.columns
        )

        # Select only the predictions corresponding to the requested forecasting horizon
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
                "model_path": "ibm-granite/granite-timeseries-patchtsmixer",
                "revision": "main",
                "config": {
                    "context_length": 8,
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
    """Pad or truncate sequences to match the required context length."""
    batch_size, original_seq_len, n_dims = data.shape

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


def _frame2numpy(data):
    """Convert pandas MultiIndex DataFrame to numpy array."""
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr


def _same_index(data):
    """Ensure all time series have the same index length."""
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(
        lambda x: x.equals(data.iloc[0])
    ).all(), "All series must have the same index"
    return data.iloc[0], len(data.iloc[0])


class PyTorchDataset(Dataset):
    """Dataset class for PyTorch models in sktime forecasting."""

    def __init__(self, y, context_length, prediction_length):
        """Initialize the dataset.

        Parameters
        ----------
        y : ndarray or pandas DataFrame
            The time series data with shape (n_sequences, n_timestamps, n_dims).
        context_length : int
            The length of past values used as input.
        prediction_length : int
            The length of future values to predict.
        """
        self.context_length = context_length
        self.prediction_length = prediction_length

        # Convert y to numpy array if it has a MultiIndex
        if isinstance(y.index, pd.MultiIndex):
            self.y = _frame2numpy(y)
        else:
            self.y = np.expand_dims(y.values, axis=0)

        self.n_sequences, self.n_timestamps, _ = self.y.shape
        self.single_length = (
            self.n_timestamps - self.context_length - self.prediction_length + 1
        )

    def __len__(self):
        """Return the total number of samples available in the dataset."""
        return self.single_length * self.n_sequences

    def __getitem__(self, i):
        """Return a single sample from the dataset."""
        from torch import tensor

        m = i % self.single_length  # Determines position within sequence
        n = i // self.single_length  # Determines which sequence

        past_values = self.y[n, m : m + self.context_length, :]
        future_values = self.y[
            n,
            m + self.context_length : m + self.context_length + self.prediction_length,
            :,
        ]
        observed_mask = np.ones_like(past_values)

        return {
            "past_values": tensor(past_values).float(),
            "observed_mask": tensor(observed_mask).float(),
            "future_values": tensor(future_values).float(),
        }
