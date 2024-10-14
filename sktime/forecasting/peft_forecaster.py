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
    input_model : sktime._BaseGlobalForecaster or transformers.PreTrainedModel
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
        input_model,
        peft_config,
    ):
        # self.input_model is a sktime global forecasting object
        self.input_model = input_model
        # user passed in peft_config
        self.peft_config = peft_config
        # make a deep copy of the input model as we do not want to change
        # anything from the original model
        self.model_copy = deepcopy(self.input_model)

        # locate and grab the underlying torch model
        self.base_model = _check_model_input(self.input_model)
        # check to make sure that the peft config is valid
        config = _check_peft_config(peft_config)

        # create the `PeftModel` from the base_model
        self.peft_model = get_peft_model(self.base_model, config)
        # self.model = self.peft_model
        # no major use for the model_copy right now, can be omitted
        self.model_copy._peft_model = self.peft_model

        # this portion of the code is commented out because it
        # should only exist in the fit instance..
        # uncomment this code if you want to inspect the newforecaster
        # parameters etc..
        self.newforecaster = type(self.model_copy)(**self.model_copy.get_params())
        self.newforecaster._peft_model = self.peft_model
        super().__init__()

    def _fit(self, fh, X, y):
        # New object initialized with type()
        self.newforecaster = type(self.input_model)(**self.input_model.get_params())
        self.newforecaster.peft_model = self.peft_model
        # print(self.newforecaster.peft_model)
        self.newforecaster.fit(fh=fh, X=X, y=y)
        # print(self.newforecaster.peft_model)
        return self

    def _predict(self, fh, X, y):
        y_pred = self.model_copy.predict(fh=fh, X=X, y=y)

        return y_pred


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


def add_adapter(self, peft_config):
    """Add adapter function."""
    # ensure that self.peft_model exists
    if not hasattr(self, "peft_model"):
        raise Exception
    else:
        self.peft_model.add_adapter(peft_config)
