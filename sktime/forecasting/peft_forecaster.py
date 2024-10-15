"""PeFT Forecaster module for applying PeFT Methods on sktime global forecasters."""

from copy import deepcopy

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
    forecaster : sktime._BaseGlobalForecaster or transformers.PreTrainedModel
        or nn.Module, required
        The base model used for Peft. If a user is passing in a sktime
        global forecaster,  the underlying torch module must be an
        available attribute accessible by the `PeftForecaster`

    peft_config : PeftConfig, required

    sequence_length : int, optional
        default = 3

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
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
    }

    def __init__(
        self,
        forecaster,
        peft_config,
    ):
        # self.input_model is a sktime global forecasting object
        self.forecaster = forecaster
        # user passed in peft_config
        self.peft_config = peft_config
        # make a deep copy of the input model as we do not want to change
        # anything from the original model
        self.forecaster_copy = deepcopy(self.forecaster)

        # locate and grab the underlying torch model
        self.base_model = _check_model_input(self.forecaster_copy)
        # check to make sure that the peft config is valid
        self.config = _check_peft_config(peft_config)

        # create the `PeftModel` from the base_model
        self.peft_model = get_peft_model(self.base_model, self.config)
        # self.model = self.peft_model
        super().__init__()

    def _fit(self, fh, X, y):
        original_params = self.forecaster_copy.get_params()
        del original_params["peft_model"]

        self.new_forecaster = type(self.forecaster_copy)(
            **original_params, peft_model=self.peft_model
        )
        self.new_forecaster.fit(fh=fh, X=X, y=y)
        return self

    def _predict(self, fh, X, y):
        y_pred = self.new_forecaster.predict(fh=fh, X=X, y=y)

        return y_pred


def _check_model_input(forecaster):
    """Check if the passed model is valid for the PeftForecaster."""
    if isinstance(forecaster, _BaseGlobalForecaster):
        if not hasattr(forecaster, "model"):
            raise AttributeError(
                "For sktime deep learning forecasters,"
                " an attribute named 'model' containing "
                "the underlying torch model is required."
            )
        else:
            base_model = forecaster.model
            if isinstance(base_model, (Module, PreTrainedModel)):
                return base_model
            else:
                raise TypeError(
                    "Expected a nn.Module or a PreTrainedModel "
                    "but found"
                    f" {type(forecaster).__name__}"
                )
    else:
        if isinstance(forecaster, (Module, PreTrainedModel)):
            return forecaster
        else:
            raise TypeError(
                "Expected a nn.Module or a PreTrainedModel"
                f" but found {type(forecaster).__name__}"
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


def add_adapter(self, peft_config):
    """Add adapter function."""
    # ensure that self.peft_model exists
    if not hasattr(self, "peft_model"):
        raise Exception
    else:
        self.peft_model.add_adapter(peft_config)
