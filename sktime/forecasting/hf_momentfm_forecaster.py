"""Interface for the momentfm deep learning time series forecaster."""

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster  # , ForecastingHorizon

if _check_soft_dependencies(["momentfm", "torch"], severity="none"):
    import momentfm
    from torch.nn import MSELoss
    from torch.utils.data import Dataset


class MomentFMForecaster(_BaseGlobalForecaster):
    """
    Interface for forecasting with the deep learning time series model momentfm.

    MomentFM is a collection of open source foundation models for the general
    purpose of time series analysis. The Moment Foundation Model is a pre-trained
    model that is capable of accomplishing various time series tasks, such as:
        - Long Term Forecasting
        - Short Term Forecasting

    This interface with MomentFM focuses on the forecasting task, in which the
    foundation model uses a user fine tuned 'forecasting head' to predict h steps ahead.
    This model does NOT have zero shot capabilities and requires fine-tuning
    to achieve performance on user inputted data.

    For more information: see
    https://github.com/moment-timeseries-foundation-model/moment

    For information regarding licensing and use of the momentfm model please visit:
    https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md

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

    dropout : float
        Dropout value of the model. Values range between [0.0, 1.0]
        Default = 0.1

    head_dropout : float
        Dropout value of the forecasting head. Values range between [0.0, 1.0]
        Default = 0.1

    batch_size : int
        size of batches to train the model on

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
        Default is 'google/flan-t5-large'.

    config : dict, default = {}
        If desired, user can pass in a config detailing all momentfm parameters
        that they wish to set in dictionary form, so that parameters do not need
        to be individually set. If a parameter inside a config is a
        duplicate of one already passed in individually, it will be overwritten.
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
        dropout=0.1,
        head_dropout=0.1,
        batch_size=8,
        epochs=0.1,
        max_lr=1e-4,
        device="gpu",
        train_val_split=0.2,
        transformer_backbone="google/flan-t5-large",
        config=None,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.freeze_encoder = freeze_encoder
        self.freeze_embedder = freeze_embedder
        self.freeze_head = freeze_head
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_lr = max_lr
        self.device = device
        self.train_val_split = train_val_split
        self.transformer_backbone = transformer_backbone
        self.config = config
        self._config = config if config is not None else {}
        self.criterion = MSELoss()

    def _fit(self, y, X, fh):
        # from torch.optim import Adam
        from torch.utils.data import DataLoader

        self._pretrained_model_name_or_path = self._config.getattr(
            "pretrained_model_name_or_path", self.pretrained_model_name_or_path
        )
        self._freeze_encoder = self._config.getattr(
            "freeze_encoder", self.freeze_encoder
        )
        self._freeze_embedder = self._config.getattr(
            "freeze_embedder", self.freeze_embedder
        )
        self._freeze_head = self._config.getattr("freeze_head", self.freeze_head)
        self._dropout = self._config.getattr("dropout", self.dropout)
        self._head_dropout = self._config.getattr("head_dropout", self.head_dropout)
        self._device = self._config.getattr("device", self.device)
        self._transformer_backbone = self._config.getattr(
            "transformer_backbone", self.transformer_backbone
        )
        # in the case the config contains 'forecasting_horizon', we'll set
        # fh as that, otherwise we override it using the fh param
        self._fh = self._config.getattr("forecasting_horizon", None)
        if self._fh is None:
            self._fh = max(fh.to_relative(self.cutoff))
        self._freeze_head = self._config.getattr("freeze_head", self.freeze_head)
        train_val_split = int(len(y) * (1 - self.train_val_split))

        self._model = momentfm.MOMENTPipeline.from_pretrained(
            self._pretrained_model_name_or_path,
            model_kwargs={
                "task_name": "forecasting",
                "dropout": self._dropout,
                "head_dropout": self._head_dropout,
                "freeze_encoder": self._freeze_encoder,
                "freeze_embedder": self._freeze_embedder,
                "freeze_head": self._freeze_head,
                "device": self._device,
                "transformer_backbone": self._transformer_backbone,
                "forecasting_horizon": self._fh,
            },
        )

        train_dataset = momentPytorchDataset(
            y=y[:train_val_split],
            X=X[:train_val_split],
            fh=self._fh,
            seq_len=512,  # fixed due to momentfm model pre-requisite
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = momentPytorchDataset(
            y=y[train_val_split:],
            X=X[train_val_split:],
            fh=self._fh,
            seq_len=512,  # fixed due to momentfm model pre-requisite
        )

        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=True
        )

        # returning dataloaders to pass pre-commit checks, WIP
        return train_dataloader, val_dataloader

    def _predict(sefl, X, fh):
        pass


class momentPytorchDataset(Dataset):
    """Customized Pytorch dataset for the momentfm model."""

    def __init__(self, y, fh, seq_len, X=None):
        self.y = y
        self.X = X if X is not None else X
        self.seq_len = seq_len
        self.fh = fh

    def __len__(self):
        """Return length of dataset."""
        return len(self.y) - self.seq_len - self.fh + 1

    def __getitem__(self, i):
        """Return dataset items from index i."""
        from torch import from_numpy, ones

        hist_end = i + self.seq_len
        pred_end = i + self.seq_len + self.fh

        input_mask = ones(self.y.shape[0])
        historical_y = from_numpy(self.y.iloc[i:hist_end].values)
        future_y = from_numpy(self.y.iloc[hist_end:pred_end].values)
        if self.X is not None:
            historical_x = from_numpy(self.X.iloc[i:hist_end].values)
            future_x = from_numpy(self.X.iloc[hist_end:pred_end])
        return {
            "future_y": future_y,
            "future_x": future_x,
            "historical_y": historical_y,
            "historical_x": historical_x,
            "input_mask": input_mask,
        }
