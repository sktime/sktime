"""Peft Adapter module for applying Peft Methods on sktime global forecasters."""

from copy import deepcopy

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster

if _check_soft_dependencies("torch", severity="none"):
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies(["peft", "transformers"], severity="none"):
    # from transformers import AutoConfig, Trainer, TrainingArguments
    from peft import PeftConfig, PeftType, get_peft_model

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

__author__ = ["julian-fong"]


class PeftForecaster(BaseForecaster):
    """Implementation of the Peft Forecaster for sktime deep learning forecasters.

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

    >>> from peft import LoraConfig
    >>> config = LoraConfig(
    ...     r = 8,
    ...     lora_alpha = 32,
    ...     target_modules = ["k_proj", "v_proj", "q_proj"],
    ... )
    >>> model = forecaster.model
    >>> peftforecaster = PeftForecaster(model, config, forecaster)
    >>> peftforecaster.fit(y) # doctest: +SKIP
    >>> y_pred = peftforecaster.predict(fh = fh) # doctest: +SKIP
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_dependencies": ["transformers", "torch", "peft"],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.Series",
        "capability:insample": False,
        "capability:pred_int:insample": False,
    }

    def __init__(self, model, peft_config, sktime_forecaster=None):
        self.model = model
        self.peft_config = _check_peft_config(peft_config)
        self.sktime_forecaster = sktime_forecaster
        self.peft_model = get_peft_model(self.model, self.peft_config)
        super().__init__()

    def _fit(self, y, fh, X=None):
        if not self.sktime_forecaster:
            raise ValueError("A sktime based forecaster must be passed in to call fit")

        original_params = deepcopy(self.sktime_forecaster.get_params())
        original_params["model_path"] = self.peft_model
        self.forecaster = type(self.sktime_forecaster)(**original_params)
        self.forecaster.fit(y=y, fh=fh, X=X)

        return self

    def _predict(self, fh, X=None):
        y_pred = self.forecaster.predict(fh=fh, X=X)

        return y_pred

    def set_peft_model(self, model):
        self.peft_model = model

    def get_peft_model(self):
        return self.peft_model

    def get_fitted_forecaster(self):
        return self.forecaster


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


# class sktimePeftAdapterMixin:
#     """
#     Parameter efficient fine tuning methods for global forecasters in sktime.

#     Parameters
#     ----------
#     model : Pytorch Model
#         The model to apply peft methods on.
#     peft_config : PeftConfig
#         The configuration to use for the peft model.
#     sktime_forecaster : sktime GlobalForecaster/Forecaster
#         The sktime forecaster to extract the model from if passed.
#     model_attribute :
#         the name of the attribute where the Pytorch model is located.
#         If passed, sktime_forecaster must be passed as well.
#         default : "model"
#     """

#     def __init__(
#         self,
#         model,
#         peft_config,
#         sktime_forecaster = None,
#         model_attribute = "model",
#     ):
#         self.model = model
#         self.peft_config = peft_config
#         self.sktime_forecaster = sktime_forecaster
#         self.model_attribute = model_attribute
#         super().__init__()

#     def add_adapter(self, peft_config):
#         """Add adapter function."""
#         # ensure that self.peft_model exists
#         if not hasattr(self, "peft_model"):
#             raise Exception
#         else:
#             self.peft_model.add_adapter(peft_config)
