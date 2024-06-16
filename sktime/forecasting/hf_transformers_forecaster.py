"""Adapter for using huggingface transformers for forecasting."""

from copy import deepcopy
import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""

        pass


if _check_soft_dependencies("transformers", severity="none"):
    from transformers import AutoConfig, Trainer, TrainingArguments
    from transformers import InformerModel, AutoformerModel, TimeSeriesTransformerModel

from sktime.forecasting.base import BaseForecaster

__author__ = ["benheid"]


class HFTransformersForecaster(BaseForecaster):
    """
    Forecaster that uses a huggingface model for forecasting.

    This forecaster fetches the model from the huggingface model hub.
    Note, this forecaster is in an experimental state. It is currently only
    working for Informer, Autoformer, and TimeSeriesTransformer.

    Parameters
    ----------
    model_path : str or preTrained Model
        Path to the huggingface model to use for forecasting. Should be in the
        format "huggingface/{model-name}". Alternatively, a pre-loaded
        Huggingface model object can be passed.
    fit_strategy : str, default="minimal"
        Strategy to use for fitting the model. Can be "minimal" or "full"
    validation_split : float, default=0.2
        Fraction of the data to use for validation
    config : dict, default={}
        Configuration to use for the model. See the `transformers`
        documentation for details.
    training_args : dict, default={}
        Training arguments to use for the model. See `transformers.TrainingArguments`
        for details. Note that the `output_dir` argument is required.
    compute_metrics : list, default=None
        List of metrics to compute during training. See `transformers.Trainer`
        for details.
    deterministic : bool, default=False
        Whether the predictions should be deterministic or not.
    callbacks : list, default=[]
        List of callbacks to use during training. See `transformers.Trainer`

    Supported Architectures
    -----------------------
    - Informer
    - Autoformer
    - TimeSeriesTransformer

    Examples
    --------
    >>> from transformers import InformerModel
    >>> from sktime.forecasting.hf_transformers_forecaster import ( #noqa
    ...     HFTransformersForecaster, #noqa
    ... ) #noqa
    >>> from sktime.datasets import load_airline

    ... # Using a model path
    >>> forecaster = HFTransformersForecaster(
    ...     model_path="huggingface/informer-tourism-monthly",
    ...     fit_strategy="minimal",
    ...     training_args={
    ...         "num_train_epochs": 1,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 32,
    ...     },
    ...     config={
    ...         "lags_sequence": [1, 2, 3],
    ...         "context_length": 2,
    ...         "prediction_length": 4,
    ...         "use_cpu": True,
    ...         "label_length": 2,
    ...     },
    ...     deterministic=True,
    ...     # doctest: +SKIP
    ... )
    >>> forecaster.fit(y)
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP

    ... # Using a pre-loaded model object
    >>> model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly")
    >>> forecaster_with_model_obj = HFTransformersForecaster(
    ...     model_path=model,
    ...     fit_strategy="minimal",
    ...     training_args={
    ...         "num_train_epochs": 1,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 32,
    ...     },
    ...     config={
    ...         "lags_sequence": [1, 2, 3],
    ...         "context_length": 2,
    ...         "prediction_length": 4,
    ...         "use_cpu": True,
    ...         "label_length": 2,
    ...     },
    ...     deterministic=True,
    ...     # doctest: +SKIP
    ... )
    >>> forecaster_with_model_obj.fit(y)
    >>> y_pred = forecaster_with_model_obj.predict(fh) # doctest: +SKIP
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_dependencies": ["transformers", "torch"],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.Series",
        "capability:insample": False,
        "capability:pred_int:insample": False,
    }

    SUPPORTED_ARCHITECTURES = ["Informer", "Autoformer", "TimeSeriesTransformer"]

    MODEL_CLASSES = {
        "Informer": InformerModel,
        "Autoformer": AutoformerModel,
        "TimeSeriesTransformer": TimeSeriesTransformerModel,
    }

    def __init__(
        self,
        model_path,
        fit_strategy="minimal",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        deterministic=False,
        callbacks=None,
    ):
        super().__init__()
        self.model_path = model_path
        self.fit_strategy = fit_strategy
        self.validation_split = validation_split
        self.config = config
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
        self.compute_metrics = compute_metrics
        self._compute_metrics = compute_metrics
        self.deterministic = deterministic
        self.callbacks = callbacks
        self._callbacks = callbacks

    def _fit(self, y, X, fh):
        if isinstance(self.model_path, str):
            # Load model and extract config
            config = AutoConfig.from_pretrained(self.model_path)
        else:
            # Use the provided model object
            config = self.model_path.config

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self._config)
        _config["num_dynamic_real_features"] = X.shape[-1] if X is not None else 0
        _config["num_static_real_features"] = 0
        _config["num_dynamic_real_features"] = 0
        _config["num_static_categorical_features"] = 0
        _config["num_time_features"] = 0 if X is None else X.shape[-1]

        if hasattr(config, "feature_size"):
            del _config["feature_size"]

        if fh is not None:
            _config["prediction_length"] = max(
                *(fh.to_relative(self._cutoff)._values + 1),
                _config["prediction_length"],
            )

        config = config.from_dict(_config)

        if isinstance(self.model_path, str):
            model_class = self.MODEL_CLASSES.get(config.architectures[0], None)
            if model_class is None:
                raise ValueError(
                    f"The model architecture {config.architectures[0]} is not supported"
                )

            # Load model with the updated config
            self.model, info = model_class.from_pretrained(
                self.model_path,
                config=config,
                output_loading_info=True,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = self.model_path
            info = {"mismatched_keys": []}

        # Freeze all loaded parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Clamp all loaded parameters to avoid NaNs due to large values
        for param in self.model.model.parameters():
            param.clamp_(-1000, 1000)

        # Reinit the weights of all layers that have mismatched sizes
        for key, _, _ in info["mismatched_keys"]:
            _model = self.model
            for attr_name in key.split(".")[:-1]:
                _model = getattr(_model, attr_name)
            _model.weight = torch.nn.Parameter(
                _model.weight.masked_fill(_model.weight.isnan(), 0.001),
                requires_grad=True,
            )

        if self.validation_split is not None:
            split = int(len(y) * (1 - self.validation_split))

            train_dataset = PyTorchDataset(
                y[:split],
                config.context_length + max(config.lags_sequence),
                X=X[:split] if X is not None else None,
                fh=config.prediction_length,
            )

            eval_dataset = PyTorchDataset(
                y[split:],
                config.context_length + max(config.lags_sequence),
                X=X[split:] if X is not None else None,
                fh=config.prediction_length,
            )
        else:
            train_dataset = PyTorchDataset(
                y,
                config.context_length + max(config.lags_sequence),
                X=X,
                fh=config.prediction_length,
            )
            eval_dataset = None

        training_args = TrainingArguments(**self._training_args)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=self._callbacks,
        )

        self.trainer.train()

    def _predict(self, fh, X=None):
        predictions = self.trainer.predict(self.trainer.eval_dataset)._predictions
        return predictions


class PyTorchDataset(Dataset):
    """PyTorch dataset for time series forecasting."""

    def __init__(self, y, context_length, X=None, fh=None):
        self.y = y
        self.context_length = context_length
        self.X = X
        self.fh = fh
        self.indices = []
        for i in range(len(y) - context_length - fh + 1):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        y_slice = self.y[i : i + self.context_length]
        y_slice = np.array(y_slice)
        if self.X is not None:
            X_slice = self.X[i : i + self.context_length]
            X_slice = np.array(X_slice)
        else:
            X_slice = np.empty((self.context_length, 0))
        y_target = self.y[i + self.context_length : i + self.context_length + self.fh]
        y_target = np.array(y_target)
        return {"input": y_slice, "target": y_target, "exog": X_slice}
