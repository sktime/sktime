"""Interface for the momentfm deep learning time series forecaster."""

import warnings

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster
from sktime.split import temporal_train_test_split

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
    https://huggingface.co/AutonLab/MOMENT-1-large

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

    seq_len : int
        length of sequences or length of historical values that are passed
        to the model for training at each time point. the momentfm model requires
        sequence lengths to be 512 exactly, so if less, padding will be used.
        If the sequence length is > 512, it will be reduced to 512.
        default = 512

    batch_size : int
        size of batches to train the model on
        default = 8

    eval_batch_size : int or "all"
        size of batches to evaluate the model on. If the string "all" is
        specified, then we process the entire validation set as a single batch
        default = 8

    epochs : int
        Number of epochs to fit tune the model on
        default = 1

    max_lr : float
        Maximum learning rate that the learning rate scheduler will use

    device : str
        torch device to use gpu else cpu

    pct_start : float
        percentage of total iterations where the learning rate rises during
        one epoch

    max_norm : float
        gradient clipping value

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
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "python_dependencies": ["momentfm", "torch", "tqdm"],
        "capability:global_forecasting": False,
    }

    def __init__(
        self,
        pretrained_model_name_or_path="AutonLab/MOMENT-1-large",
        freeze_encoder=True,
        freeze_embedder=True,
        freeze_head=False,
        dropout=0.1,
        head_dropout=0.1,
        seq_len=512,
        batch_size=8,
        eval_batch_size=8,
        epochs=1,
        max_lr=1e-4,
        device="gpu",
        pct_start=0.3,
        max_norm=5.0,
        train_val_split=0.25,
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
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.max_lr = max_lr
        self.device = device
        self.pct_start = pct_start
        self.max_norm = max_norm
        self.train_val_split = train_val_split
        self.transformer_backbone = transformer_backbone
        self.config = config
        self._config = config if config is not None else {}
        self.criterion = MSELoss()
        self._moment_seq_len = 512

    def _fit(self, fh, y, X=None):
        """Assumes y is a single or multivariate time series."""
        import torch.cuda.amp
        from torch.optim import Adam
        from torch.optim.lr_scheduler import OneCycleLR
        from torch.utils.data import DataLoader

        self._pretrained_model_name_or_path = (
            self._config["pretrained_model_name_or_path"]
            if "pretrained_model_name_or_path" in self._config.keys()
            else self.pretrained_model_name_or_path
        )
        self._freeze_encoder = (
            self._config["freeze_encoder"]
            if "freeze_encoder" in self._config.keys()
            else self.freeze_encoder
        )
        self._freeze_embedder = (
            self._config["freeze_embedder"]
            if "_freeze_embedder" in self._config.keys()
            else self.freeze_embedder
        )
        self._freeze_head = (
            self._config["freeze_head"]
            if "freeze_head" in self._config.keys()
            else self.freeze_head
        )
        self._dropout = (
            self._config["dropout"]
            if "dropout" in self._config.keys()
            else self.dropout
        )
        self._head_dropout = (
            self._config["head_dropout"]
            if "head_dropout" in self._config.keys()
            else self.head_dropout
        )
        self._transformer_backbone = (
            self._config["transformer_backbone"]
            if "transformer_backbone" in self._config.keys()
            else self.transformer_backbone
        )
        self._criterion = (
            self._config["criterion"]
            if "criterion" in self._config.keys()
            else self.criterion
        )
        # evaluate the sequence length passed by the user
        self._seq_len = (
            self._config["seq_len"]
            if "seq_len" in self._config.keys()
            else self.seq_len
        )
        if self._seq_len > self._moment_seq_len:
            warnings.warn(
                f"length of {self._seq_len} was found which is greater than 512. "
                "The most recent 512"
                " values will be used when fitting.",
                stacklevel=2,
            )
            self._seq_len = 512
        # in the case the config contains 'forecast_horizon', we'll set
        # fh as that, otherwise we override it using the fh param
        self._fh_config = (
            self._config["forecast_horizon"]
            if "forecast_horizon" in self._config.keys()
            else self.fh
        )
        self._device = (
            self._config["device"] if "device" in self._config.keys() else self.device
        )
        # check availability of user specified device
        self._device = _check_device(self._device)

        cur_epoch = 0
        max_epoch = self.epochs
        if fh is not None:
            self._fh_input = max(fh.to_relative(self.cutoff))
        self._fh = self._fh_input if fh is not None else self._fh_config
        self._model = momentfm.MOMENTPipeline.from_pretrained(
            self._pretrained_model_name_or_path,
            model_kwargs={
                "task_name": "forecasting",
                "dropout": self._dropout,
                "head_dropout": self._head_dropout,
                "freeze_encoder": self._freeze_encoder,
                "freeze_embedder": self._freeze_embedder,
                "seq_len": 512,  # forced to be hard coded
                "freeze_head": self._freeze_head,
                "device": self._device,
                "transformer_backbone": self._transformer_backbone,
                "forecast_horizon": self._fh,
            },
        )
        self._model.init()
        self._y_cols = y.columns
        self._y_shape = y.values.shape

        # preparing the datasets
        y_train, y_test = temporal_train_test_split(
            y, train_size=1 - self.train_val_split, test_size=self.train_val_split
        )

        if y_train.shape[0] < 512:
            y_train = _sample_observations(y_train)
        if y_test.shape[0] < 512:
            y_test = _sample_observations(y_test)

        train_dataset = MomentPytorchDataset(
            y=y_train,
            fh=self._fh,
            seq_len=self._seq_len,
            device=self._device,
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = MomentPytorchDataset(
            y=y_test,
            fh=self._fh,
            seq_len=self._seq_len,
            device=self._device,
        )

        if self.eval_batch_size == "all":
            self._eval_batch_size = len(val_dataset)
        else:
            self._eval_batch_size = self.eval_batch_size

        val_dataloader = DataLoader(
            val_dataset, batch_size=self._eval_batch_size, shuffle=True
        )

        criterion = self._criterion
        optimizer = Adam(self._model.parameters(), lr=self.max_lr)

        # Move the model to the GPU
        self._model = self._model.to(self._device)

        # Move the loss function to the GPU
        criterion = criterion.to(self._device)

        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Create a OneCycleLR scheduler
        total_steps = len(train_dataloader) * max_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
        )

        # Gradient clipping value
        max_norm = self.max_norm

        while cur_epoch < max_epoch:
            cur_epoch = _run_epoch(
                cur_epoch,
                criterion,
                optimizer,
                scheduler,
                scaler,
                self._model,
                max_norm,
                train_dataloader,
                val_dataloader,
            )
        return self

    def _predict(self, y, fh=None, X=None):
        """Predict method to forecast timesteps into the future.

        fh must be the same length as the one used to fit the model.
        """
        from torch import from_numpy

        self._model = self._model.to(self._device)
        self._model.eval()
        # first convert it into numpy values
        y_ = y.values
        sequence_length, num_channels = y_.shape  # shape of our input to predict
        # raise warning if sequence length of y is greater than the sequence
        # length used to fit the model
        if sequence_length > self._seq_len:
            warnings.warn(
                f"length of {y.shape[0]} was found which is greater than sequence "
                "length {self._seq_len} used to fit the model. The most recent"
                f" {self._seq_len} values will be used.",
                stacklevel=2,
            )
            # only retain the most recent self._seq_len values if greater than
            # self._seq_len
            y_ = y_[-self._seq_len :, :]
            sequence_length = self._seq_len

        # transpose it to change it into (C, S) size
        y_ = y_.T
        if sequence_length < self._moment_seq_len:
            # if smaller, need to pad values
            y_ = _create_padding(
                y_, (self._moment_seq_len - sequence_length, num_channels)
            )
            input_mask = _create_mask(
                sequence_length, self._moment_seq_len - sequence_length
            )
        else:  # this means sequence_length = self._seq_len == 512
            input_mask = _create_mask(self._moment_seq_len)
        if num_channels != self._y_shape[1]:
            # Todo raise error here
            pass
        # returns a timeseriesoutput object
        y_torch_input = (
            from_numpy(y_.reshape((1, self._y_shape[1], -1))).float().to(self._device)
        )
        input_mask = input_mask.to(self._device)
        output = self._model(y_torch_input, input_mask)
        forecast_output = output.forecast
        forecast_output = forecast_output.squeeze(0)

        pred = forecast_output.detach().cpu().numpy().T

        df_pred = pd.DataFrame(pred, columns=self._y_cols)

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
        params_set = []
        params1 = {}
        params_set.append(params1)

        return params_set


def _create_padding(x, pad_shape):
    """Return zero padded tensor of size seq_len, num_cols."""
    #    For example, if num_rows = 500 and seq_len = 512
    # then x.shape[0] = 500 and pad_shape[0] = 12
    # then cat(x, zero_pad) should return (num_cols,512)
    from torch import cat, zeros

    if isinstance(x, np.ndarray):
        from torch import from_numpy

        x_ = from_numpy(x)
    else:
        x_ = x
    zero_pad = zeros(pad_shape).T  # transpose to make it size (C, 512)
    out = cat((x_, zero_pad), axis=1)
    if isinstance(x, np.ndarray):
        out = np.array(out)
    return out


def _create_mask(ones_length, zeros_length=0):
    from torch import cat, ones, zeros

    zeros_tensor = zeros(zeros_length)
    ones_tensor = ones(ones_length)

    input_mask = cat((ones_tensor, zeros_tensor))
    return input_mask


def _run_epoch(
    cur_epoch,
    criterion,
    optimizer,
    scheduler,
    scaler,
    model,
    max_norm,
    train_dataloader,
    val_dataloader,
):
    import torch.cuda.amp
    from momentfm.utils.forecasting_metrics import get_forecasting_metrics
    from tqdm import tqdm

    losses = []
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        # Move the data to the GPU
        timeseries = data["historical_y"]
        input_mask = data["input_mask"]
        forecast = data["future_y"]
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
    average_loss = np.average(losses)
    tqdm.write(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

    # Step the learning rate scheduler
    scheduler.step()
    cur_epoch += 1

    # Evaluate the model on the test split
    trues, preds, histories, losses = [], [], [], []
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_dataloader, total=len(val_dataloader)):
            # Move the data to the specified device
            timeseries = data["historical_y"]
            input_mask = data["input_mask"]
            forecast = data["future_y"]

            with torch.cuda.amp.autocast():
                output = model(timeseries, input_mask)

            loss = criterion(output.forecast, forecast)
            losses.append(loss.item())

            trues.append(forecast.detach().cpu().numpy())
            preds.append(output.forecast.detach().cpu().numpy())
            histories.append(timeseries.detach().cpu().numpy())

    losses = np.array(losses)
    average_loss = np.average(losses)
    model.train()

    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    histories = np.concatenate(histories, axis=0)

    metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction="mean")
    tqdm.write(
        f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f}"
        f"|Test MAE: {metrics.mae:.3f}"
    )

    return cur_epoch


def _check_device(device):
    mps = False
    cuda = False
    if device == "mps":
        from torch.backends.mps import is_available, is_built

        if not is_available():
            if not is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        else:
            _device = "mps"
            mps = True
    elif device == "gpu" or device == "cuda":
        from torch.cuda import is_available

        if is_available():
            _device = "cuda"
            cuda = True
    elif "cuda" in device:  # for specific cuda devices like cuda:0 etc
        from torch.cuda import is_available

        if is_available():
            _device = device
            cuda = True
    if mps or cuda:
        return _device
    else:
        _device = "cpu"

    return _device


def _sample_observations(y):
    n_total_samples = 1000
    y_sampled = y.sample(n=n_total_samples, replace=True)

    return y_sampled


class MomentPytorchDataset(Dataset):
    """Customized Pytorch dataset for the momentfm model."""

    def __init__(self, y, fh, seq_len, device):
        self.y = y
        self.moment_seq_len = 512  # forced seq_len by pre-trained model
        self.seq_len = seq_len
        self.fh = fh
        self.shape = y.shape
        self.device = device

        # code block to figure out masking sizes in case seq_len < 512
        if self.seq_len < self.moment_seq_len:
            self._pad_shape = (self.moment_seq_len - self.seq_len, self.shape[1])
            self.input_mask = (
                _create_mask(self.seq_len, self._pad_shape[0]).float().to(self.device)
            )
            # Concatenate the tensors
        elif self.seq_len > self.moment_seq_len:
            # for now if seq_len > 512 than we reduce it back to 512
            self.seq_len = 512
            self.input_mask = _create_mask(self.seq_len).float().to(self.device)
        else:
            self.input_mask = _create_mask(self.seq_len).float().to(self.device)

    def __len__(self):
        """Return length of dataset."""
        return len(self.y) - self.seq_len - self.fh + 1

    def __getitem__(self, i):
        """Return dataset items from index i."""
        # batches must be returned in format (B, C, S)
        # where B = batch_size, C = channels, S = sequence_length
        from torch import from_numpy

        hist_end = i + self.seq_len
        pred_end = i + self.seq_len + self.fh

        historical_y = (
            from_numpy(self.y.iloc[i:hist_end].values)
            .float()
            .reshape(self.y.shape[1], -1)
        )
        if self.seq_len < self.moment_seq_len:
            historical_y = _create_padding(historical_y, self._pad_shape)
        historical_y = historical_y.float().to(self.device)
        future_y = (
            (
                from_numpy(self.y.iloc[hist_end:pred_end].values)
                .float()
                .reshape(self.y.shape[1], -1)
            )
            .float()
            .to(self.device)
        )
        return {
            "future_y": future_y,
            "historical_y": historical_y,
            "input_mask": self.input_mask,
        }
