"""Adapter for using huggingface transformers for forecasting."""

from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.split import temporal_train_test_split

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies("transformers", severity="none"):
    import transformers
    from transformers import AutoConfig, Trainer, TrainingArguments

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster

__author__ = ["benheid", "geetu040", "XinyuWu"]


class HFTransformersForecaster(_BaseGlobalForecaster):
    """
    Forecaster that uses a huggingface model for forecasting.

    This forecaster fetches the model from the huggingface model hub.
    Note, this forecaster is in an experimental state. It is currently only
    working for Informer, Autoformer, and TimeSeriesTransformer.

    Parameters
    ----------
    model_path : str
        Path to the huggingface model to use for forecasting. Currently,
        Informer, Autoformer, and TimeSeriesTransformer are supported.
    fit_strategy : str, default="minimal"
        Strategy to use for fitting (fine-tuning) the model. This can be one of
        the following:
        - "minimal": Fine-tunes only a small subset of the model parameters,
          allowing for quick adaptation with limited computational resources.
        - "full": Fine-tunes all model parameters, which may result in better
          performance but requires more computational power and time.
        - "peft": Applies Parameter-Efficient Fine-Tuning (PEFT) techniques to adapt
          the model with fewer trainable parameters, saving computational resources.
          Note: If the 'peft' package is not available, a `ModuleNotFoundError` will
          be raised, indicating that the 'peft' package is required. Please install
          it using `pip install peft` to use this fit strategy.
    broadcasting: bool (default=True)
        DeprecationWarning: default value will be changed to False in v0.34.0
        multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from `predict`.
    validation_split : float, default=0.2
        Fraction of the data to use for validation
    config : dict, default={}
        Configuration to use for the model. See the `transformers`
        documentation for details.
    training_args : dict, default={}
        Training arguments to use for the model. See `transformers.TrainingArguments`
        for details.
        Note that the `output_dir` argument is required.
    compute_metrics : list, default=None
        List of metrics to compute during training. See `transformers.Trainer`
        for details.
    deterministic : bool, default=False
        Whether the predictions should be deterministic or not.
    callbacks : list, default=[]
        List of callbacks to use during training. See `transformers.Trainer`
    peft_config : peft.PeftConfig, default=None
        Configuration for Parameter-Efficient Fine-Tuning.
        When `fit_strategy` is set to "peft",
        this will be used to set up PEFT parameters for the model.
        See the `peft` documentation for details.
    no_size1_batch: bool, default=True
        drop the last batch if batch size is one.
        It's not `drop_last` of pytorch dataloader [1]_,
        it will only drop if last batch size is exactly one.
        The batch size is from training_args["per_device_train_batch_size"].
        If no training_args["per_device_train_batch_size"] passed, it's default 8 [2]_.
    try_local_files_only: bool, default=False
        Try to load config and model in `local_files_only` mode first,
        if any error raises, load again in normal mode.
        See HuggingFace offline mode for details [3]_.


    References
    ----------
    .. [1] https://pytorch.org/docs/stable/data.html
    .. [2] https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size
    .. [3] https://huggingface.co/docs/transformers/main/en/installation#offline-mode
    # noqa: E501


    Examples
    --------
    >>> from sktime.forecasting.hf_transformers_forecaster import (
    ...     HFTransformersForecaster,
    ... )
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = HFTransformersForecaster(
    ...    model_path="huggingface/autoformer-tourism-monthly",
    ...    training_args ={
    ...        "num_train_epochs": 20,
    ...        "output_dir": "test_output",
    ...        "per_device_train_batch_size": 32,
    ...    },
    ...    config={
    ...         "lags_sequence": [1, 2, 3],
    ...         "context_length": 2,
    ...         "prediction_length": 4,
    ...         "use_cpu": True,
    ...         "label_length": 2,
    ...    },
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y) # doctest: +SKIP
    >>> fh = [1, 2, 3]
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP

    >>> from sktime.forecasting.hf_transformers_forecaster import (
    ...     HFTransformersForecaster,
    ... )
    >>> from sktime.datasets import load_airline
    >>> from peft import LoraConfig
    >>> y = load_airline()
    >>> forecaster = HFTransformersForecaster(
    ...    model_path="huggingface/autoformer-tourism-monthly",
    ...    fit_strategy="peft",
    ...    training_args={
    ...        "num_train_epochs": 20,
    ...        "output_dir": "test_output",
    ...        "per_device_train_batch_size": 32,
    ...    },
    ...    config={
    ...         "lags_sequence": [1, 2, 3],
    ...         "context_length": 2,
    ...         "prediction_length": 4,
    ...         "use_cpu": True,
    ...         "label_length": 2,
    ...    },
    ...    peft_config=LoraConfig(
    ...        r=8,
    ...        lora_alpha=32,
    ...        target_modules=["q_proj", "v_proj"],
    ...        lora_dropout=0.01,
    ...    )
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y) # doctest: +SKIP
    >>> fh = [1, 2, 3]
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_dependencies": ["transformers", "torch"],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "y_inner_mtype": [
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
    }

    def __init__(
        self,
        model_path: str,
        fit_strategy="minimal",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        deterministic=False,
        callbacks=None,
        peft_config=None,
        no_size1_batch=True,
        try_local_files_only=False,
        # TODO change the default value to False in v0.34.0
        broadcasting=True,
    ):
        super().__init__()
        self.model_path = model_path
        self.fit_strategy = fit_strategy
        self.broadcasting = broadcasting
        self.validation_split = validation_split
        self.config = config
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
        if "per_device_train_batch_size" not in self._training_args.keys():
            self._training_args["per_device_train_batch_size"] = 8
        self.compute_metrics = compute_metrics
        self._compute_metrics = compute_metrics
        self._compute_metrics = compute_metrics
        self.deterministic = deterministic
        self.callbacks = callbacks
        self._callbacks = callbacks
        self.peft_config = peft_config
        self.no_size1_batch = no_size1_batch
        self.try_local_files_only = try_local_files_only

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.Series",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

        warn(
            "DeprecationWarning: The default value of the parameter "
            "broadcasting will be set to False in v0.34.0.",
            DeprecationWarning,
        )

    def _fit(self, y, X, fh):
        def try_local_files_only(f: callable):
            try:
                return f(True)
            except Exception:
                return f(False)

        def load_config(local_files_only=False):
            return AutoConfig.from_pretrained(
                self.model_path,
                local_files_only=local_files_only,
            )

        # Load model and extract config
        if self.try_local_files_only:
            config = try_local_files_only(load_config)
        else:
            config = load_config(False)

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self._config)
        _config["num_static_real_features"] = 0
        _config["num_dynamic_real_features"] = 0
        _config["num_static_categorical_features"] = 0
        _config["num_time_features"] = 0 if X is None else X.shape[-1]

        if hasattr(config, "feature_size"):
            del _config["feature_size"]

        if fh is not None:
            _config["prediction_length"] = max(
                *(fh.to_relative(self._cutoff)._values),
                _config["prediction_length"],
            )

        config = config.from_dict(_config)
        import transformers

        prediction_model_class = None
        if hasattr(config, "architectures") and config.architectures is not None:
            prediction_model_class = config.architectures[0]
        elif hasattr(config, "model_type"):
            prediction_model_class = (
                "".join(x.capitalize() for x in config.model_type.lower().split("_"))
                + "ForPrediction"
            )
        else:
            raise ValueError(
                "The model type is not inferable from the config."
                "Thus, the model cannot be loaded."
            )

        def load_model(local_files_only=False):
            model, info = getattr(transformers, prediction_model_class).from_pretrained(
                self.model_path,
                config=config,
                output_loading_info=True,
                ignore_mismatched_sizes=True,
                local_files_only=local_files_only,
            )
            return model, info

        # Load model with the updated config
        if self.try_local_files_only:
            self.model, info = try_local_files_only(load_model)
        else:
            self.model, info = load_model(False)

        # Freeze all loaded parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Clamp all loaded parameters to avoid NaNs due to large values
        for param in self.model.model.parameters():
            param.clamp_(-1000, 1000)

        # Reininit the weights of all layers that have mismatched sizes
        for key, _, _ in info["mismatched_keys"]:
            _model = self.model
            for attr_name in key.split(".")[:-1]:
                _model = getattr(_model, attr_name)
            _model.weight = torch.nn.Parameter(
                _model.weight.masked_fill(_model.weight.isnan(), 0.001),
                requires_grad=True,
            )

        if self.validation_split is not None:
            if X is None:
                y_train, y_test = temporal_train_test_split(
                    y, test_size=self.validation_split
                )
            else:
                y_train, y_test, X_train, X_test = temporal_train_test_split(
                    y, X, test_size=self.validation_split
                )

            train_dataset = PyTorchDataset(
                y_train,
                config.context_length + max(config.lags_sequence),
                X=X_train if X is not None else None,
                fh=config.prediction_length,
                batch_size=self._training_args["per_device_train_batch_size"],
                no_size1_batch=self.no_size1_batch,
            )

            eval_dataset = PyTorchDataset(
                y_test,
                config.context_length + max(config.lags_sequence),
                X=X_test if X is not None else None,
                fh=config.prediction_length,
                batch_size=self._training_args["per_device_train_batch_size"],
                no_size1_batch=self.no_size1_batch,
            )
        else:
            train_dataset = PyTorchDataset(
                y,
                config.context_length + max(config.lags_sequence),
                X=X if X is not None else None,
                fh=config.prediction_length,
                batch_size=self._training_args["per_device_train_batch_size"],
                no_size1_batch=self.no_size1_batch,
            )

            eval_dataset = None

        training_args = deepcopy(self._training_args)
        training_args["label_names"] = ["future_values"]
        training_args = TrainingArguments(**training_args)

        if self.fit_strategy == "minimal":
            if len(info["mismatched_keys"]) == 0:
                return  # No need to fit
        elif self.fit_strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        elif self.fit_strategy == "peft":
            if _check_soft_dependencies("peft", severity="none"):
                from peft import get_peft_model
            else:
                raise ModuleNotFoundError(
                    f"Error in {self.__class__.__name__}: 'peft' module not found. "
                    "'peft' is a soft dependency and not included "
                    "in the base sktime installation. "
                    "To use this functionality, please install 'peft' by running: "
                    "`pip install peft` or `pip install sktime[dl]`. "
                    "To install all soft dependencies, "
                    "run: `pip install sktime[all_extras]`"
                )
            peft_config = deepcopy(self.peft_config)
            self.model = get_peft_model(self.model, peft_config)
        else:
            raise ValueError("Unknown fit strategy")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=self._callbacks,
        )
        trainer.train()

    def _predict(self, fh, X=None, y=None):
        if self.deterministic:
            transformers.set_seed(42)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        self.model.eval()
        hist_y = y if self._global_forecasting else self._y

        if not isinstance(hist_y.index, pd.MultiIndex):
            hist_y = _to_multiindex(hist_y)
            converted_to_multiindex = True
        else:
            converted_to_multiindex = False

        if X is not None:
            if not isinstance(X.index, pd.MultiIndex):
                X = _to_multiindex(X)
            if not self._global_forecasting:
                if not isinstance(self._X.index, pd.MultiIndex):
                    _X = _to_multiindex(self._X)
                else:
                    _X = self._X

        hist = _frame2numpy(hist_y).squeeze(2)

        if X is not None:
            if not self._global_forecasting:
                hist_x = _frame2numpy(_X)
                x_ = _frame2numpy(X)
            else:
                len_levels = len(X.index.names)
                ins_levels = list(range(len_levels - 1))
                # groupby instances levels, get the history exogenous data
                # of each instances by slicing the time index
                hist_x = _frame2numpy(
                    X.groupby(level=ins_levels).apply(
                        lambda x: x.droplevel(ins_levels).iloc[
                            : -self.model.config.prediction_length
                        ]
                    )
                )
                # groupby instances levels, get the last prediction_length
                # of the time index as the future exogenous data
                x_ = _frame2numpy(
                    X.groupby(level=ins_levels).apply(
                        lambda x: x.droplevel(ins_levels).iloc[
                            -self.model.config.prediction_length :
                        ]
                    )
                )

            if x_.shape[1] < self.model.config.prediction_length:
                # TODO raise exception here?
                x_ = np.resize(
                    x_, (x_.shape[0], self.model.config.prediction_length, x_.shape[-1])
                )
        else:
            hist_x = np.array(
                [
                    [[]]
                    * (
                        self.model.config.context_length
                        + max(self.model.config.lags_sequence)
                    )
                ]
                * hist.shape[0]
            )
            x_ = np.array([[[]] * self.model.config.prediction_length] * hist.shape[0])

        from torch import from_numpy

        pred = self.model.generate(
            past_values=from_numpy(hist).to(self.model.dtype).to(self.model.device),
            past_time_features=from_numpy(
                hist_x[
                    :,
                    -self.model.config.context_length
                    - max(self.model.config.lags_sequence) :,
                ]
            )
            .to(self.model.dtype)
            .to(self.model.device),
            future_time_features=from_numpy(x_)
            .to(self.model.dtype)
            .to(self.model.device),
            past_observed_mask=from_numpy((~np.isnan(hist)).astype(int)).to(
                self.model.device
            ),
        )

        pred = pred.sequences.mean(dim=1).detach().cpu().numpy()

        ins = np.array(
            list(np.unique(hist_y.index.droplevel(-1)).repeat(pred.shape[1]))
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
            names=hist_y.index.names,
        )

        pred = pd.DataFrame(
            pred.flatten(),
            index=index,
            columns=hist_y.columns,
        )

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = pred.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        pred = pred.loc[dateindex]

        if converted_to_multiindex:
            pred = pd.Series(
                pred.values.squeeze(),
                index=pred.index.get_level_values(-1),
                name=pred.columns[0],
            )

        return pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

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
                "model_path": "huggingface/informer-tourism-monthly",
                "fit_strategy": "minimal",
                "validation_split": None,  # some series in CI are too short
                "training_args": {
                    "num_train_epochs": 1,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 32,
                },
                "config": {
                    "lags_sequence": [1, 2, 3],
                    "context_length": 2,
                    "prediction_length": 4,
                },
                "deterministic": True,
                "try_local_files_only": True,
            },
            {
                "model_path": "huggingface/autoformer-tourism-monthly",
                "fit_strategy": "minimal",
                "validation_split": None,  # some series in CI are too short
                "training_args": {
                    "num_train_epochs": 1,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 32,
                },
                "config": {
                    "lags_sequence": [1, 2, 3],
                    "context_length": 2,
                    "prediction_length": 4,
                    "label_length": 2,
                },
                "deterministic": True,
                "try_local_files_only": True,
            },
        ]

        if _check_soft_dependencies("peft", severity="none"):
            from peft import LoraConfig

            test_params.append(
                {
                    "model_path": "huggingface/autoformer-tourism-monthly",
                    "fit_strategy": "peft",
                    "training_args": {
                        "num_train_epochs": 1,
                        "output_dir": "test_output",
                        "per_device_train_batch_size": 32,
                    },
                    "config": {
                        "lags_sequence": [1, 2, 3],
                        "context_length": 2,
                        "prediction_length": 4,
                        "label_length": 2,
                    },
                    "peft_config": LoraConfig(
                        r=2,
                        lora_alpha=8,
                        target_modules=["q_proj"],
                        lora_dropout=0.01,
                    ),
                    "deterministic": True,
                    "try_local_files_only": True,
                }
            )
        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in test_params]
        params_no_broadcasting = [
            dict(p, **{"broadcasting": False}) for p in test_params
        ]
        return params_broadcasting + params_no_broadcasting


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(
        self,
        y: pd.DataFrame,
        seq_len: int,
        fh=None,
        X: pd.DataFrame = None,
        batch_size=8,
        no_size1_batch=True,
    ):
        if not isinstance(y.index, pd.MultiIndex):
            self.y = np.array(y.values, dtype=np.float32).reshape(1, len(y), 1)
            self.X = (
                np.array(X.values, dtype=np.float32).reshape(
                    (1, len(X), len(X.columns))
                )
                if X is not None
                else X
            )
        else:
            self.y = _frame2numpy(y)
            self.X = _frame2numpy(X) if X is not None else X

        self._num, self._len, _ = self.y.shape
        self.fh = fh
        self.seq_len = seq_len
        self._len_single = self._len - self.seq_len - self.fh + 1
        self.batch_size = batch_size
        self.no_size1_batch = no_size1_batch

    def __len__(self):
        """Return length of dataset."""
        true_length = self._num * max(self._len_single, 0)
        if self.no_size1_batch and true_length % self.batch_size == 1:
            return true_length - 1
        else:
            return true_length

    def __getitem__(self, i):
        """Return data point."""
        from torch import tensor

        m = i % self._len_single
        n = i // self._len_single
        hist_y = tensor(self.y[n, m : m + self.seq_len, :]).float().flatten()
        futu_y = (
            tensor(self.y[n, m + self.seq_len : m + self.seq_len + self.fh, :])
            .float()
            .flatten()
        )
        if self.X is not None:
            exog_data = tensor(
                self.X[n, m + self.seq_len : m + self.seq_len + self.fh, :]
            ).float()
            hist_exog = tensor(self.X[n, m : m + self.seq_len, :]).float()
        else:
            exog_data = tensor([[]] * self.fh)
            hist_exog = tensor([[]] * self.seq_len)

        return {
            "past_values": hist_y,
            "past_time_features": hist_exog,
            "future_time_features": exog_data,
            "past_observed_mask": (~hist_y.isnan()).to(int),
            "future_values": futu_y,
        }


def _same_index(data):
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(
        lambda x: x.equals(data.iloc[0])
    ).all(), "All series must has the same index"
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data):
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr


def _to_multiindex(data, index_name="h0", instance_name="h0_0"):
    res = pd.DataFrame(
        data.values,
        index=pd.MultiIndex.from_product(
            [[instance_name], data.index], names=[index_name, data.index.name]
        ),
        columns=[data.name] if isinstance(data, pd.Series) else data.columns,
    )
    return res
