"""Interface for the momentfm deep learning time series forecaster."""

# from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster  # , ForecastingHorizon

# if _check_soft_dependencies(["momentfm", "torch"], severity="none"):
#     import momemtfm


class MomentFMForecaster(BaseForecaster):
    """
    Interface for forecasting with the deep learning time series model momentfm.

    MomentFM is a collection of open source foundation models for the general
    purpose of time series analysis. The Moment Foundation Model is a pre-trained
    model that is capable of accomplishing various time series tasks, such as:
        - Long Term Forecasting
        - Short Term Forecasting
        - Classification
        - Imputation
        - Anomaly Detection

    This interface with MomentFM focuses on the forecasting task, in which the
    foundation model uses a user fine tuned 'forecasting head' to predict h steps ahead.
    This model does NOT have zero shot capabilities and requires fine-tuning
    to achieve performance on user inputted data.

    For more information: see
    https://github.com/moment-timeseries-foundation-model/moment

    pretrained_model_name_or_path : str
        Path to the pretrained Momentfm model. Default is AutonLab/MOMENT-1-large

    freeze_encoder : bool
        Selection of whether or not to freeze the weights of the encoder
        Default = True

    freeze_embedder : bool
        Selection whether or not to freeze the patch embedding layer
        Default = True

    freeze_head : bool
        Selection whether or not to freeze the forecasting head.
        Recommendation is that the linear forecasting head must be trained
        Default = False

    d_model : int
        Dimensionality of the inputs into the transformer model. If d_model is None
        then d_model is extracted from a specified transformer_backbone

    dropout : float
        Dropout value of the model. Values range between [0.0, 1.0]
        Default = 0.1

    head_dropout : float
        Dropout value of the forecasting head. Values range between [0.0, 1.0]
        Default = 0.1

    epochs : int
        Number of epochs to fit tune the model on

    max_lr : float
        Maximum learning rate that the learning rate scheduler will use

    device : str
        torch device to use gpu else cpu

    train_val_split : float
        float value between 0 and 1 to determine portions of training
        and validation splits

    transformer_backbone : str
        d_model of a pre-trained transformer model to use. See
        SUPPORTED_HUGGINGFACE_MODELS to specify valid models to use.
        Default is 'google/flan-t5-large'. Used if d_model is None
    """

    _tags = {
        "scitype:y": "both",
        "authors": ["julian-fong"],
        "maintainers": ["julian-fong"],
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "python_dependencies": ["momentfm", "torch"],
    }

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        freeze_encoder: True,
        freeze_embedder: True,
        freeze_head: False,
        d_model: None,
        dropout=0.1,
        head_dropout=0.1,
        epochs=0.1,
        max_lr=1e-4,
        device="gpu",
        train_val_split=0.2,
        transformer_backbone="google/flan-t5-large",
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.freeze_encoder = freeze_encoder
        self.freeze_embedder = freeze_embedder
        self.freeze_head = freeze_head
        self.d_model = d_model
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.epochs = epochs
        self.max_lr = max_lr
        self.device = device
        self.train_val_split = train_val_split
        self.transformer_backbone = transformer_backbone
