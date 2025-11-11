# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Adapter for using the huggingface PatchTST for forecasting."""

__author__ = [
    "julian-fong",
    "geetu040",
    "Yuqi Nie",
    "Nam H. Nguyen",
    "Phanwadee Sinthong",
    "Jayant Kalagnanam",
]


import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
Dataset = _safe_import("torch.utils.data.Dataset")

PatchTSTConfig = _safe_import("transformers.PatchTSTConfig")
PatchTSTForPrediction = _safe_import("transformers.PatchTSTForPrediction")
PatchTSTModel = _safe_import("transformers.PatchTSTModel")
Trainer = _safe_import("transformers.Trainer")
TrainingArguments = _safe_import("transformers.TrainingArguments")


class PatchTSTForecaster(_BaseGlobalForecaster):
    """Interface for the PatchTST forecaster.

    This forecaster interfaces the Huggingface library's PatchTST model for
    time series forecasting. The model was originally designed by Yuqi Nie,
    Nam H. Nguyen, Phanwadee Sinthong and Jayant Kalagnanam.
    It utilizes a transformer model architecture and splits the
    time series data into patches that are then processed by the model.
    Visit [1] for more information on the model architecture and its authors.
    For tips on how to construct your own PatchTST config, see [2].

    The PatchTST forecaster can be used in three ways:

    1) Full training via a new model initialized from a config or a loaded model
    with pre-trained weights
    2) Minimal fine-tuning with a pre-trained model and an altered config
    3) Zero-shot forecasting with a pre-trained model

    For more details, please visit the `fit_strategy` parameter

    Parameters
    ----------
    model_path : str or PatchTSTModel, optional
        Path to the Huggingface model to use for global forecasting. If
        model_path is passed, the remaining model config parameters will be
        ignored except for specific training or dataset parameters.
        This has 3 options:

        - model id to an online pretrained PatchTST Model hosted on HuggingFace
        - A path or url to a saved configuration JSON file
        - A path to a *directory* containing a configuration file saved

        using the ``~PretrainedConfig.save_pretrained`` method
        or the ``~PreTrainedModel.save_pretrained`` method

    fit_strategy : str, values = ["full","minimal","zero-shot"], default = "full"
        String to set the fit_strategy of the model.

        - This strategy is used to create and train a new model from scratch
        (pre-pretraining) or to update all of the weights in a pre-trained model
        (also known as full fine-tuning). If `fit_strategy` is set to `full`,
        requires either the `model_path` parameter or the `config`` parameter
        to be passed in, but not both. If only `config` is passed, it will
        initialize an new model with untrained weights with the specified config
        arguments. If only `model_path` is passed, it will fine-tune ALL of the
        pre-trained weights of the model.

        - If `fit_strategy` is set to "minimal" requires both the `model_path`
        and `config` parameter. We will use the `model_path` and the specified
        `config` to compare the weight shapes of the passed pre-trained model
        to those in the config. If there are weight size mismatches, the model
        will reinitialize new weights to match the weight shapes inside the `config`.
        The `y` argument will then be fit to fine-tune the model. In the case where
        there are no newly initialized weights (i.e the config weight shapes match
        the pretrained model weight shapes), it will behave the same as the "full"
        strategy where only the `model_path` is passed in.

        - If `fit_strategy` is set to "zero-shot", requires only the `model_path`
        parameter. It will load the model via the `fit` function with the argument
        `model_path` and ignore any passed `y`.

    validation_split : float, optional, default = 0.2
        Fraction of the data to use for validation.

    config : dict, optional, default = {}
        A config dict specifying parameters to initialize an full
        PatchTST model. Missing parameters in the config will be automatically
        replaced by their default values. See the PatchTSTConfig config on
        huggingface for more details.
        Note: if `prediction_length` is passed as in larger than the passed `fh`
        in the `fit` function, the `prediction_length` will be used to train the
        model. If `prediction_length` is passed as in smaller than the passed
        `fh` in the `fit` function, the passed `fh` will be used to train the
        model.

    training_args : dict, optional, default = None
        Training arguments to use for the model. If this is passed,
        the remaining applicable training arguments will be ignored
    compute_metrics : list or function, default = None
        List of metrics or function to use during training
    callbacks: list or function, default = None
        List of callbacks or callback function to use during training

    References
    ----------
    [1] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
        Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
        Paper: https://arxiv.org/abs/2211.14730
    [2] HuggingFace PatchTST Page:
        https://huggingface.co/docs/transformers/en/model_doc/patchtst

    Examples
    --------
    >>> #Example with a new model initialized from config only
    >>> from sktime.forecasting.patch_tst import PatchTSTForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = PatchTSTForecaster(
    ... config = {
    ...     "patch_length": 1,
    ...      "context_length": 2,
    ...      "patch_stride": 1,
    ...      "d_model": 64,
    ...      "num_attention_heads": 2,
    ...      "ffn_dim": 32,
    ...      "head_dropout": 0.3,
    ...    },
    ...    training_args = {
    ...         "output_dir":"test_output",
    ...         "overwrite_output_dir":True,
    ...         "learning_rate":1e-4,
    ...         "num_train_epochs":1,
    ...         "per_device_train_batch_size":16,
    ...    }
    ... ) #initialize an full model
    >>> forecaster.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP

    >>> #Example full fine-tuning with a pre-trained model
    >>> from sktime.forecasting.patch_tst import PatchTSTForecaster
    >>> import pandas as pd
    >>> dataset_path = pd.read_csv(
    ...     "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    ...     ).drop(columns = ["date"]
    ... )
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = StandardScaler()
    >>> scaler.set_output(transform="pandas") # doctest: +SKIP
    >>> scaler = scaler.fit(dataset_path.values) # doctest: +SKIP
    >>> df = scaler.transform(dataset_path)  # doctest: +SKIP
    >>> df.columns = dataset_path.columns # doctest: +SKIP
    >>> forecaster = PatchTSTForecaster(
    ...     model_path="namctin/patchtst_etth1_forecast",
    ...     fit_strategy = "full",
    ...     training_args = {
    ...         "output_dir":"test_output",
    ...         "overwrite_output_dir":True,
    ...         "learning_rate":1e-4,
    ...         "num_train_epochs":1,
    ...         "per_device_train_batch_size":16,
    ...     }
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y = df, fh = list(range(1,4))) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP

    >>> #Example of minimal fine-tuning with a pre-trained model and an altered config
    >>> from sktime.forecasting.patch_tst import PatchTSTForecaster
    >>> import pandas as pd
    >>> dataset_path = pd.read_csv(
    ...     "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    ...     ).drop(columns = ["date"]
    ... )
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = StandardScaler()
    >>> scaler.set_output(transform="pandas") # doctest: +SKIP
    >>> scaler = scaler.fit(dataset_path.values) # doctest: +SKIP
    >>> df = scaler.transform(dataset_path) # doctest: +SKIP
    >>> df.columns = dataset_path.columns # doctest: +SKIP
    >>> forecaster = PatchTSTForecaster(
    ...     model_path="namctin/patchtst_etth1_forecast",
    ...     config = {
    ...         "patch_length": 8,
    ...         "context_length": 512,
    ...         "patch_stride": 8,
    ...         "d_model": 128,
    ...         "num_attention_heads": 2,
    ...         "ffn_dim": 512,
    ...         "head_dropout": 0.3,
    ...         "prediction_length": 64
    ...     },
    ...     fit_strategy = "minimal",
    ...     training_args = {
    ...         "output_dir":"test_output",
    ...         "overwrite_output_dir":True,
    ...         "learning_rate":1e-4,
    ...         "num_train_epochs":1,
    ...         "per_device_train_batch_size":16,
    ...     }
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y = df, fh = list(range(1,63))) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP

    >>> #Example with a pre-trained model to do zero-shot forecasting
    >>> from sktime.forecasting.patch_tst import PatchTSTForecaster
    >>> import pandas as pd
    >>> dataset_path = pd.read_csv(
    ...     "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    ...     ).drop(columns = ["date"]
    ... )
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = StandardScaler()
    >>> scaler.set_output(transform="pandas")  # doctest: +SKIP
    >>> scaler = scaler.fit(dataset_path.values)  # doctest: +SKIP
    >>> df = scaler.transform(dataset_path) # doctest: +SKIP
    >>> df.columns = dataset_path.columns # doctest: +SKIP
    >>> forecaster = PatchTSTForecaster(
    ...     model_path="namctin/patchtst_etth1_forecast",
    ...     fit_strategy = "zero-shot",
    ...     training_args = {
    ...         "output_dir":"test_output",
    ...         "overwrite_output_dir":True,
    ...         "learning_rate":1e-4,
    ...         "num_train_epochs":1,
    ...         "per_device_train_batch_size":16,
    ...     }
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y = df, fh = [1,2,3,4,5]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP
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
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "authors": [
            "julian-fong",
            "geetu040",
            "Yuqi Nie",
            "Nam H. Nguyen",
            "Phanwadee Sinthong",
            "Jayant Kalagnanam",
        ],
        "maintainers": ["julian-fong"],
        "python_dependencies": ["transformers", "torch", "accelerate"],
        "capability:global_forecasting": True,
        "tests:vm": True,
    }

    def __init__(
        self,
        # model variables except for forecast_columns
        model_path=None,
        fit_strategy="full",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
    ):
        self.model_path = model_path
        self.fit_strategy = fit_strategy
        # dataset and training parameters
        self.validation_split = validation_split
        self.config = config
        self._config = self.config if self.config else {}
        self.training_args = training_args
        self._training_args = self.training_args if self.training_args else {}
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks

        self._config = self.config if self.config else {}
        super().__init__()
        if self.fit_strategy not in ["full", "minimal", "zero-shot"]:
            raise ValueError("unexpected fit_strategy passed in argument")

        if self.model_path is None and self.fit_strategy != "full":
            raise ValueError(f"model_path={model_path} requires fit_strategy=='full'")

    def _fit(self, y, fh, X=None):
        """Fits the model.

        Parameters
        ----------
        y : pandas DataFrame
            pandas dataframe containing single or multivariate data

        fh : Forecasting Horizon object
            used to determine forecasting horizon for predictions

        Returns
        -------
        self : a reference to the object
        """
        if isinstance(self.model_path, PatchTSTModel):
            self.model = self.model_path
            config = self.model.config
        else:
            if self.model_path:
                config = PatchTSTConfig.from_pretrained(self.model_path)
            else:
                config = PatchTSTConfig()

            # Update config with user provided config
            _config = config.to_dict()
            _config.update(self._config)

            # Update config with model specific parameters
            _config["num_input_channels"] = len(y.columns)
            if fh is not None:
                _config["prediction_length"] = max(
                    *(fh.to_relative(self._cutoff)._values + 1),
                    _config["prediction_length"],
                )

            config = config.from_dict(_config)

            if not self.model_path:
                self.model = PatchTSTForPrediction(config=config)
            else:
                # Load model with the passed config if it is given
                self.model, info = PatchTSTForPrediction.from_pretrained(
                    self.model_path,
                    config=config,
                    output_loading_info=True,
                    ignore_mismatched_sizes=True,
                )

            if self.fit_strategy == "zero-shot":
                if len(info["mismatched_keys"]) > 0 or len(info["missing_keys"]) > 0:
                    raise ValueError(
                        "Fit strategy is 'zero-shot', but the model weights in the"
                        "configuration are mismatched or missing compared to the "
                        "pre-trained model. This happens because of a changed "
                        "configuration."
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

        y_train, y_test = temporal_train_test_split(
            y, train_size=1 - self.validation_split, test_size=self.validation_split
        )
        train_dataset = PyTorchDataset(
            y_train,
            context_length=self.model.config.context_length,
            prediction_length=self.model.config.prediction_length,
        )
        if self.validation_split > 0.0:
            eval_dataset = PyTorchDataset(
                y_test,
                context_length=self.model.config.context_length,
                prediction_length=self.model.config.prediction_length,
            )
        else:
            eval_dataset = None

        # Get Training Configuration
        training_args = TrainingArguments(**self._training_args)
        # Get the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        # Train the model
        trainer.train()

        return self

    def _predict(self, y, X=None, fh=None):
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
            If not passed in _fit, guaranteed to be passed here. If using a pre-trained
            model, ensure that the prediction_length of the model matches the passed fh.
        y : sktime time series object, required
            single or multivariate data to compute forecasts on.

        Returns
        -------
        y_pred : sktime time series object
            pandas DataFrame
        """
        if y is None:
            y = self._y
        if fh is None:
            fh = self.fh_
        else:
            fh = fh.to_relative(self.cutoff)
        y_columns = y.columns
        y_index_names = list(y.index.names)
        # multi-index conversion
        if isinstance(y.index, pd.MultiIndex):
            _y = _frame2numpy(y)
        else:
            _y = np.expand_dims(y.values, axis=0)

        _y = torch.tensor(_y).float().to(self.model.device)

        if _y.shape[1] > self.model.config.context_length:
            _y = _y[:, -self.model.config.context_length :, :]

        # in the case where the context_length of the pre-trained model is larger
        # than the context_length of the model
        self.model.eval()
        y_pred = self.model(_y).prediction_outputs
        pred = y_pred.detach().cpu().numpy()

        if isinstance(y.index, pd.MultiIndex):
            ins = np.array(list(np.unique(y.index.droplevel(-1)).repeat(pred.shape[1])))
            ins = [ins[..., i] for i in range(ins.shape[-1])] if ins.ndim > 1 else [ins]

            idx = (
                ForecastingHorizon(range(1, pred.shape[1] + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values.tolist()
                * pred.shape[0]
            )
            index = pd.MultiIndex.from_arrays(
                ins + [idx],
                names=y.index.names,
            )
        else:
            index = (
                ForecastingHorizon(range(1, pred.shape[1] + 1))
                .to_absolute(self._cutoff)
                ._values
            )

        df_pred = pd.DataFrame(
            # batch_size * num_timestams, n_cols
            pred.reshape(-1, pred.shape[-1]),
            index=index,
            columns=y_columns,
        )

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = df_pred.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        df_pred = df_pred.loc[dateindex]
        df_pred.index.names = y_index_names
        return df_pred

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
        import platform

        os = platform.system()
        if os == "Darwin":
            if _check_soft_dependencies("torch", severity="none"):
                import torch

                torch.backends.mps.is_available = lambda: False
            else:
                pass

        params_set = []
        params1 = {
            "config": {
                "patch_length": 2,
                "context_length": 4,
                "patch_stride": 2,
                "d_model": 32,
                "num_attention_heads": 1,
                "ffn_dim": 16,
                "head_dropout": 0.3,
                "prediction_length": 2,
            },
            "training_args": {
                "output_dir": "test_output",
                "overwrite_output_dir": True,
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 16,
            },
            "validation_split": 0.0,
        }
        params_set.append(params1)
        params2 = {
            "config": {
                "patch_length": 1,
                "context_length": 2,
                "patch_stride": 1,
                "d_model": 64,
                "num_attention_heads": 2,
                "ffn_dim": 32,
                "head_dropout": 0.3,
                "prediction_length": 2,
            },
            "training_args": {
                "output_dir": "test_output",
                "overwrite_output_dir": True,
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 16,
            },
            "validation_split": 0.0,
        }
        params_set.append(params2)

        return params_set


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


# copied from the PytorchDataset module from hf_transformers_forecaster.py
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

        return {
            "past_values": tensor(past_values).float(),
            "future_values": tensor(future_values).float(),
        }
