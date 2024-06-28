"""Adapter for using huggingface transformers for forecasting."""

from copy import deepcopy

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

        pass


if _check_soft_dependencies("transformers", severity="none"):
    import transformers
    from transformers import AutoConfig, Trainer, TrainingArguments

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

__author__ = ["benheid"]


class HFTransformersForecaster(BaseForecaster):
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
        Strategy to use for fitting the model. Can be "minimal" or "full"
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
        self._compute_metrics = compute_metrics
        self.deterministic = deterministic
        self.callbacks = callbacks
        self._callbacks = callbacks

    def _fit(self, y, X, fh):
        # Load model and extract config
        config = AutoConfig.from_pretrained(self.model_path)

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
                *(fh.to_relative(self._cutoff)._values + 1),
                _config["prediction_length"],
            )
            _config["prediction_length"] = int(
                np.max(fh.to_relative(self._cutoff)._values)
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
                "The model type is not inferrable from the config."
                "Thus, the model cannot be loaded."
            )
        # Load model with the updated config
        self.model, info = getattr(
            transformers, prediction_model_class
        ).from_pretrained(
            self.model_path,
            config=config,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
        )

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
                    y, X, test_size=self.validation_split
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
            )

            eval_dataset = PyTorchDataset(
                y_test,
                config.context_length + max(config.lags_sequence),
                X=X_test if X is not None else None,
                fh=config.prediction_length,
            )
        else:
            train_dataset = PyTorchDataset(
                y,
                config.context_length + max(config.lags_sequence),
                X=X if X is not None else None,
                fh=config.prediction_length,
            )

            eval_dataset = None

        training_args = deepcopy(self.training_args)
        training_args["label_names"] = ["future_values"]
        training_args = TrainingArguments(**training_args)

        if self.fit_strategy == "minimal":
            if len(info["mismatched_keys"]) == 0:
                return  # No need to fit
        elif self.fit_strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
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

    def _predict(self, fh, X=None):
        if self.deterministic:
            transformers.set_seed(42)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        self.model.eval()
        from torch import from_numpy

        if isinstance(self._y.index, pd.MultiIndex):
            lens = self._y.groupby(
                level=list(range(len(self._y.index.levels) - 1))
            ).apply(lambda x: len(x))
            assert (lens == lens.iloc[0]).all(), "All series must has the same length"
            hist = self._y.values.reshape((-1, lens.iloc[0]))
        else:
            hist = self._y.values.reshape((1, -1))
        if X is not None:
            if isinstance(self._y.index, pd.MultiIndex):
                hist_x = np.array(self._X.values, dtype=np.float32).reshape(
                    (-1, lens.iloc[0], self._X.shape[-1])
                )
                lens = X.groupby(
                    level=list(range(len(self._y.index.levels) - 1))
                ).apply(lambda x: len(x))
                assert (
                    lens == lens.iloc[0]
                ).all(), "All series must has the same length"
                x_ = np.array(X.values, dtype=np.float32).reshape(
                    (-1, lens.iloc[0], self._X.shape[-1])
                )
                if x_.shape[1] < self.model.config.prediction_length:
                    # TODO raise exception here?
                    x_ = np.resize(
                        x_,
                        (
                            x_.shape[0],
                            self.model.config.prediction_length,
                            x_.shape[-1],
                        ),
                    )
            else:
                hist_x = self._X.values.reshape((1, -1, self._X.shape[-1]))
                x_ = X.values.reshape((1, -1, self._X.shape[-1]))
                if x_.shape[1] < self.model.config.prediction_length:
                    # TODO raise exception here?
                    x_ = np.resize(
                        x_, (1, self.model.config.prediction_length, x_.shape[-1])
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

        if isinstance(self._y.index, pd.MultiIndex):
            pred = pred.sequences.mean(dim=1).detach().cpu().numpy()

            ins = np.array(
                list(np.unique(self._y.index.droplevel(-1)).repeat(pred.shape[1]))
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
                names=self._y.index.names,
            )

            pred = pd.DataFrame(
                pred.flatten(),
                index=index,
                columns=self._y.columns,
            )

            absolute_horizons = fh.to_absolute_index(self.cutoff)
            dateindex = pred.index.get_level_values(-1).map(
                lambda x: x in absolute_horizons
            )

            pred = pred.loc[dateindex]
        else:
            pred = pred.sequences.mean(dim=1).detach().cpu().numpy().T

            pred = pd.Series(
                pred.reshape((-1,)),
                index=ForecastingHorizon(range(1, len(pred) + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values,
                name=self._y.name,
            )
            pred = pred.loc[fh.to_absolute(self.cutoff)._values]
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
        return [
            {
                "model_path": "huggingface/informer-tourism-monthly",
                "fit_strategy": "minimal",
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
            },
            {
                "model_path": "huggingface/autoformer-tourism-monthly",
                "fit_strategy": "minimal",
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
            },
        ]


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y: pd.DataFrame, seq_len: int, fh=None, X: pd.DataFrame = None):
        if not isinstance(y.index, pd.MultiIndex):
            self.y = np.array(y.values, dtype=np.float32).reshape(len(y), 1)
            self.X = (
                np.array(X.values, dtype=np.float32).reshape(
                    (len(X.columns), len(X), 1)
                )
                if X is not None
                else X
            )
        else:
            lens = y.groupby(level=list(range(len(y.index.levels) - 1))).apply(
                lambda x: len(x)
            )
            assert (lens == lens.iloc[0]).all(), "All series must has the same length"
            self.y = np.array(y.values, dtype=np.float32).reshape((lens.iloc[0], -1))
            self.X = (
                np.array(X.values, dtype=np.float32).T.reshape(
                    (len(X.columns), lens.iloc[0], -1)
                )
                if X is not None
                else X
            )

        self._len, self._num = self.y.shape
        self.fh = fh
        self.seq_len = seq_len

    def __len__(self):
        """Return length of dataset."""
        return self._num * max(self._len - self.seq_len - self.fh + 1, 0)

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor

        m = i % (self._len - self.seq_len - self.fh + 1)
        n = int((i - m) / self._len)
        hist_y = tensor(self.y[m : m + self.seq_len, n]).float()
        if self.X is not None:
            exog_data = (
                tensor(self.X[:, m + self.seq_len : m + self.seq_len + self.fh, n])
                .float()
                .T
            )
            hist_exog = tensor(self.X[:, m : m + self.seq_len, n]).float().T
        else:
            exog_data = tensor([[]] * self.fh)
            hist_exog = tensor([[]] * self.seq_len)

        return {
            "past_values": hist_y,
            "past_time_features": hist_exog,
            "future_time_features": exog_data,
            "past_observed_mask": (~hist_y.isnan()).to(int),
            "future_values": from_numpy(
                self.y[m + self.seq_len : m + self.seq_len + self.fh, n]
            ).float(),
        }
