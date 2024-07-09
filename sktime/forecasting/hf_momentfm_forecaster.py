"""Interface for the momentfm deep learning time series forecaster."""

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster  # , ForecastingHorizon

if _check_soft_dependencies(["momentfm", "torch"], severity="none"):
    import momentfm
    from torch.nn import MSELoss
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""

        pass


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

    forecasting_horizon : int
        Number of time steps to forecast ahead, leave this as None if user
        wishes to pass in a fh object instead inside the fit function
        default = None

    batch_size : int
        size of batches to train the model on
        default = 8

    epochs : int
        Number of epochs to fit tune the model on
        default = 1

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
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "python_dependencies": ["momentfm", "torch"],
        "capability:global_forecasting": False,
    }

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        freeze_encoder: True,
        freeze_embedder: True,
        freeze_head: False,
        dropout=0.1,
        head_dropout=0.1,
        forecasting_horizon=None,
        batch_size=8,
        epochs=1,
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
        self.forecasting_horizon = forecasting_horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_lr = max_lr
        self.device = device
        self.train_val_split = train_val_split
        self.transformer_backbone = transformer_backbone
        self.config = config
        self._config = config if config is not None else {}
        self.criterion = MSELoss()

    def _fit(self, y, fh):
        """Assumes y is a single or multivariate time series."""
        import torch.cuda.amp

        # from momentfm.utils.forecasting_metrics import get_forecasting_metrics
        from torch.optim import Adam
        from torch.optim.lr_scheduler import OneCycleLR
        from torch.utils.data import DataLoader
        from tqdm import tqdm

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
        self._fh = self._config.getattr("forecasting_horizon", self.forecasting_horizon)
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
            fh=self._fh,
            seq_len=512,  # fixed due to momentfm model pre-requisite
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = momentPytorchDataset(
            y=y[train_val_split:],
            fh=self._fh,
            seq_len=512,  # fixed due to momentfm model pre-requisite
        )

        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=True
        )

        # returning dataloaders to pass pre-commit checks, WIP
        criterion = MSELoss()
        optimizer = Adam(self._model.parameters, lr=1e-4)
        if self.device == "gpu" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        cur_epoch = 0
        max_epoch = self.epochs

        # Move the model to the GPU
        model = self._model.to(device)

        # Move the loss function to the GPU
        criterion = criterion.to(device)

        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Create a OneCycleLR scheduler
        max_lr = 1e-4
        total_steps = len(train_dataloader) * max_epoch
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3
        )

        # Gradient clipping value
        max_norm = 5.0

        while cur_epoch < max_epoch:
            losses = []
            for timeseries, forecast, input_mask in tqdm(
                train_dataloader, total=len(train_dataloader)
            ):
                # Move the data to the GPU
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)
                forecast = forecast.float().to(device)

                with torch.cuda.amp.autocast():
                    output = model(timeseries, input_mask)

                loss = criterion(output.forecast, forecast)

                # Scales the loss for mixed precision training
                scaler.scale(loss).backward()

                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                losses.append(loss.item())

            losses = np.array(losses)
            # average_loss = np.average(losses)
            # print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

            # Step the learning rate scheduler
            scheduler.step()
            cur_epoch += 1

            # Evaluate the model on the test split
            trues, preds, histories, losses = [], [], [], []
            model.eval()
            with torch.no_grad():
                for timeseries, forecast, input_mask in tqdm(
                    val_dataloader, total=len(val_dataset)
                ):
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(device)
                    input_mask = input_mask.to(device)
                    forecast = forecast.float().to(device)

                    with torch.cuda.amp.autocast():
                        output = model(timeseries, input_mask)

                    loss = criterion(output.forecast, forecast)
                    losses.append(loss.item())

                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(output.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())

            losses = np.array(losses)
            # average_loss = np.average(losses)
            model.train()

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)

            # metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction="mean")

            # print(
            #     f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f} |
            # Test MAE: {metrics.mae:.3f}"
            # )
        return self

    def _predict(self, X, fh):
        pass


class momentPytorchDataset(Dataset):
    """Customized Pytorch dataset for the momentfm model."""

    def __init__(self, y, fh, seq_len):
        self.y = y
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
        return {
            "future_y": future_y,
            "historical_y": historical_y,
            "input_mask": input_mask,
        }
