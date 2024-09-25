"""PeFT Forecaster module for applying PeFT Methods on sktime global forecasters."""

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    from torch.nn import Module
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies(["peft", "transformers"], severity="none"):
    # from transformers import AutoConfig, Trainer, TrainingArguments
    from peft import PeftConfig, PeftType, get_peft_model
    from transformers import PreTrainedModel

    peft_configs = [config.value for config in list(PeftType)]
    SUPPORTED_ADAPTER_CONFIGS = [
        peft_type
        for peft_type in peft_configs
        if peft_type
        not in [
            "P_TUNING",
            "PREFIX_TUNING",
            "MULTITASK_PROMPT_TUNING",
            "ADAPTION_PROMPT",
            "PROMPT_TUNING",
        ]
    ]

from sktime.forecasting.base import _BaseGlobalForecaster  # , ForecastingHorizon

__author__ = ["julian-fong"]


class PeftForecaster(_BaseGlobalForecaster):
    """Parameter efficient fine tuning methods for global forecasters in sktime.

    Parameters
    ----------
    model : sktime._BaseGlobalForecaster or transformers.PreTrainedModel
        or nn.Module, required
        The base model used for Peft. If a user is passing in a sktime
        global forecaster,  the underlying torch module must be an
        available attribute accessible by the `PeftForecaster`

    peft_config : PeftConfig, required

    sequence_length : int, optional
        default = 3

    training_args : dict, optional

    compute_metrics : callable, optional
        default = None

    callbacks : callable, optional

    datacollator : callable, optional

    broadcasting : bool,
        default=False
        if True, multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from ``predict``.

    validation_split : float in (0,1)
        default = None,

    """

    _tags = {
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "handles-missing-data": False,
        "y_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd-multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd-multiindex_hier",
        ],
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
    }

    def __init__(
        self,
        model,
        peft_config,
        sequence_length=3,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
        datacollator=None,
        broadcasting=None,
        validation_split=None,
    ):
        self.sequence_length = sequence_length
        self.training_args = training_args
        self._training_args = training_args if training_args else {}
        self.compute_metrics = compute_metrics
        self.datacollator = datacollator
        self.callbacks = callbacks
        self.broadcasting = broadcasting
        self.validation_split = validation_split

        model = _check_model_input(model)
        self.base_model = model
        config = _check_peft_config(peft_config)
        self.model = get_peft_model(model, config)

        super().__init__()

    def _fit(self, fh, X, y):
        peft_model = get_peft_model(self.model, self.config)
        return peft_model

    def _predict(self, fh, X, y):
        pass


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


def _check_model_input(model):
    """Check if the passed model is valid for the PeftForecaster."""
    if isinstance(model, _BaseGlobalForecaster):
        if not hasattr(model, "model"):
            raise AttributeError(
                "For sktime deep learning forecasters,"
                " an attribute named 'model' containing "
                "the underlying torch model is required."
            )
        else:
            base_model = model.model
            if isinstance(base_model, (Module, PreTrainedModel)):
                return base_model
            else:
                raise TypeError(
                    "Expected a nn.Module or a PreTrainedModel "
                    "but found"
                    f" {type(model).__name__}"
                )
    else:
        if isinstance(model, (Module, PreTrainedModel)):
            return model
        else:
            raise TypeError(
                "Expected a nn.Module or a PreTrainedModel"
                f" but found {type(model).__name__}"
            )


def _check_peft_config(config):
    """Check if the passed config is valid for the Peft Forecaster."""
    if not isinstance(config, PeftConfig):
        raise TypeError("Expected a PeftConfig, but found" f" {type(config).__name__}")
    else:
        if config.peft_type.value not in SUPPORTED_ADAPTER_CONFIGS:
            raise ValueError(
                f"{config.peft_type.value} is not a supported"
                " peft type. Please pass in a value that is part"
                f" of the list {SUPPORTED_ADAPTER_CONFIGS}"
            )
        else:
            return config


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
        observed_mask = np.ones_like(past_values)

        return {
            "past_values": tensor(past_values).float(),
            "observed_mask": tensor(observed_mask).float(),
            "future_values": tensor(future_values).float(),
        }
