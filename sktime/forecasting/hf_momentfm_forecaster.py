"""Interface for the momentfm deep learning time series forecaster."""

import warnings

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.split import temporal_train_test_split

if _check_soft_dependencies("torch", severity="none"):
    from torch.cuda import empty_cache
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""

        pass


if _check_soft_dependencies("accelerate", severity="none"):
    pass

if _check_soft_dependencies("transformers", severity="none"):
    from sktime.libs.momentfm import MOMENTPipeline


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
        default = 32

    eval_batch_size : int or "all"
        size of batches to evaluate the model on. If the string "all" is
        specified, then we process the entire validation set as a single batch
        default = 32

    epochs : int
        Number of epochs to fit tune the model on
        default = 1

    max_lr : float
        Maximum learning rate that the learning rate scheduler will use
        default = 1e-4

    device : str
        torch device to use
        default = "auto"
        If set to auto, it will automatically use whatever device that
        `accelerate` detects.

    pct_start : float
        percentage of total iterations where the learning rate rises during
        one epoch
        default = 0.3

    max_norm : float
        Float value used to clip gradients during training
        default = 5.0

    train_val_split : float
        float value between 0 and 1 to determine portions of training
        and validation splits
        default = 0.2

    transformer_backbone : str
        d_model of a pre-trained transformer model to use. See
        SUPPORTED_HUGGINGFACE_MODELS to specify valid models to use.
        Default is 'google/flan-t5-large'.

    config : dict, default = {}
        If desired, user can pass in a config detailing all momentfm parameters
        that they wish to set in dictionary form, so that parameters do not need
        to be individually set. If a parameter inside a config is a
        duplicate of one already passed in individually, it will be overwritten.

    criterion : criterion, default = torch.nn.MSELoss
        Criterion to use during training.

    return_model_to_cpu : bool, default = False
        After fitting and training, will return the `momentfm` model to the cpu.

    References
    ----------
    Paper: https://arxiv.org/abs/2402.03885
    Github: https://github.com/moment-timeseries-foundation-model/moment/tree/main

    Examples
    --------
    >>> from sktime.forecasting.hf_momentfm_forecaster import MomentFMForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = MomentFMForecaster(seq_len = 2)
    >>> forecaster.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict(y = y) # doctest: +SKIP
    """

    _tags = {
        "scitype:y": "both",
        "authors": ["julian-fong"],
        "maintainers": ["julian-fong"],
        "handles-missing-data": False,
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "python_dependencies": [
            "torch",
            "tqdm",
            "huggingface-hub",
            # "momentfm",
            "accelerate",
            "transformers",
        ],
        "capability:global_forecasting": True,
        "python_version": ">= 3.10",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:pred_int": False,
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
        batch_size=32,
        eval_batch_size=32,
        epochs=1,
        max_lr=1e-4,
        device="auto",
        pct_start=0.3,
        max_norm=5.0,
        train_val_split=0.2,
        transformer_backbone="google/flan-t5-large",
        criterion=None,
        config=None,
        return_model_to_cpu=False,
    ):
        super().__init__()
        from torch.nn import MSELoss

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
        self.criterion = criterion
        self._criterion = self.criterion if self.criterion else MSELoss()
        self._moment_seq_len = 512
        self.return_model_to_cpu = return_model_to_cpu

    def _fit(self, fh, y, X=None):
        """Assumes y is a single or multivariate time series."""
        from accelerate import Accelerator
        from torch.optim import Adam
        from torch.optim.lr_scheduler import OneCycleLR
        from torch.utils.data import DataLoader

        # keep a copy of y in case y is None in predict
        self._y = y
        self._y_index = self._y.index
        self._y_cols = self._y.columns
        self._y_shape = self._y.values.shape

        self._pretrained_model_name_or_path = self._config.get(
            "pretrained_model_name_or_path", self.pretrained_model_name_or_path
        )
        self._freeze_encoder = self._config.get("freeze_encoder", self.freeze_encoder)
        self._pretrained_model_name_or_path = self._config.get(
            "_pretrained_model_name_or_path", self._pretrained_model_name_or_path
        )
        self._freeze_embedder = self._config.get(
            "freeze_embedder", self.freeze_embedder
        )
        self._freeze_head = self._config.get("freeze_head", self.freeze_head)
        self._dropout = self._config.get("dropout", self.dropout)
        self._head_dropout = self._config.get("head_dropout", self.head_dropout)
        self._transformer_backbone = self._config.get(
            "transformer_backbone", self.transformer_backbone
        )
        self._criterion = self._config.get("criterion", self._criterion)
        self._seq_len = self._config.get("seq_len", self.seq_len)

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

        self._fh_config = self._config.get("forecast_horizon", None)

        # device initialization
        self._device = self._config.get("device", self.device)

        # check availability of user specified device
        self._device = _check_device(self._device)
        # initialize accelerator
        accelerator = Accelerator()
        if self._device == "auto":
            self._device = accelerator.device

        cur_epoch = 0
        max_epoch = self.epochs
        if fh is not None:
            self._fh_input = max(fh.to_relative(self.cutoff))
        self._fh = self._fh_input if fh is not None else self._fh_config
        # self._model_fh is guaranteed to be an int value
        self._model_fh = self._fh
        # revert self._fh back to fh to pass checks
        self._fh = fh

        self.model = MOMENTPipeline.from_pretrained(
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
                "forecast_horizon": self._model_fh,
            },
        )
        self.model.init()
        # preparing the datasets
        y_train, y_test = temporal_train_test_split(
            y, train_size=1 - self.train_val_split, test_size=self.train_val_split
        )

        train_dataset = MomentPytorchDataset(
            y=y_train,
            fh=self._model_fh,
            seq_len=self._seq_len,
            device=self._device,
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        if not y_test.empty:
            val_dataset = MomentPytorchDataset(
                y=y_test,
                fh=self._model_fh,
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
        else:
            val_dataloader = None

        criterion = self._criterion
        optimizer = Adam(self.model.parameters(), lr=self.max_lr)
        # Enable mixed precision training

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

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = (
            accelerator.prepare(
                self.model, optimizer, train_dataloader, val_dataloader, scheduler
            )
        )

        while cur_epoch < max_epoch:
            cur_epoch = _run_epoch(
                cur_epoch,
                accelerator,
                criterion,
                optimizer,
                scheduler,
                self.model,
                max_norm,
                train_dataloader,
                val_dataloader,
            )

        if self.return_model_to_cpu:
            self.model.to("cpu")
            empty_cache()

        return self

    def _predict(self, y, X=None, fh=None):
        """Predict method to forecast timesteps into the future.

        fh should not be passed here and
        must be the same length as the one used to fit the model.
        """
        # use y values from fit if y is None in predict
        if y is None:
            y = self._y

        index = self._fh.to_absolute_index(self.cutoff)
        from torch import from_numpy

        self.model = self.model.to(self._device)
        self.model.eval()
        y_index_names = list(y.index.names)
        if isinstance(y.index, pd.MultiIndex):
            y_ = _frame2numpy(y)
        else:
            y_ = np.expand_dims(y.values, axis=0)

        num_instances, sequence_length, num_channels = (
            y_.shape
        )  # shape of our input to predict
        # raise warning if sequence length of y is greater than the sequence
        # length used to fit the model
        if sequence_length > self._seq_len:
            warnings.warn(
                f"Sequence length of {sequence_length} was found which is greater "
                "than sequence "
                f"length {self._seq_len} used to fit the model. The most recent"
                f" {self._seq_len} values will be used.",
                stacklevel=2,
            )
            # truncate code
            # only retain the most recent self._seq_len values if greater than
            # self._seq_len
            y_ = y_[:, -self._seq_len :, :]
            sequence_length = self._seq_len
        if sequence_length < self._moment_seq_len:
            # if smaller, need to pad values
            y_ = _create_padding(
                y_,
                (num_instances, self._moment_seq_len - sequence_length, num_channels),
            )
            input_mask = _create_mask(
                sequence_length, self._moment_seq_len - sequence_length
            )
        else:  # this means sequence_length = self._seq_len == 512
            input_mask = _create_mask(self._moment_seq_len)
        if num_channels != self._y_shape[1]:
            raise ValueError(
                "The number of multivariate time series "
                f"{num_channels} passed in predict does not match the "
                f"number of multivariate time series {self._y_shape[1]} "
                "used to train the model in fit."
            )
            pass
        # transpose it to change it into (C, S) size
        y_ = y_.transpose(0, 2, 1)
        # returns a timeseriesoutput object
        y_torch_input = from_numpy(y_).float().to(self._device)
        input_mask = input_mask.to(self._device)
        output = self.model(x_enc=y_torch_input, mask=input_mask)
        forecast_output = output.forecast
        # forecast_output = forecast_output.squeeze(0)

        pred = forecast_output.detach().cpu().numpy()
        # revert back to (B, S, C)
        pred = np.transpose(pred, (0, 2, 1))

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
            columns=self._y_cols,
        )

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = df_pred.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        df_pred = df_pred.loc[dateindex]
        df_pred.index.names = y_index_names

        if self.return_model_to_cpu:
            self.model.to("cpu")
            empty_cache()

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
        params1 = {"seq_len": 2, "return_model_to_cpu": True, "train_val_split": 0.0}
        params_set.append(params1)
        params2 = {
            "batch_size": 16,
            "seq_len": 2,
            "return_model_to_cpu": True,
            "train_val_split": 0.0,
        }
        params_set.append(params2)

        return params_set


def _create_padding(x, pad_shape):
    """Return zero padded tensor of size seq_len, num_cols."""
    # For example, if num_rows = 500 and seq_len = 512
    # then x.shape[0] = 500 and pad_shape[0] = 12
    # then cat(x, zero_pad) should return (num_cols,512)
    from torch import cat, zeros

    if isinstance(x, np.ndarray):
        from torch import from_numpy

        x_ = from_numpy(x)
    else:
        x_ = x

    pad_shape_dim = x_.dim()
    # create a padding of size 2 for fit()
    if pad_shape_dim == 2:
        zero_pad = zeros(pad_shape)
        axis = 0
    else:
        # if its 3 dimensions, we are in predict()
        zero_pad = zeros(pad_shape)
        axis = 1

    out = cat((x_, zero_pad), axis=axis)
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
    accelerator,
    criterion,
    optimizer,
    scheduler,
    model,
    max_norm,
    train_dataloader,
    val_dataloader,
):
    import torch.cuda.amp
    from tqdm import tqdm

    from sktime.libs.momentfm.utils.forecasting_metrics import get_forecasting_metrics

    losses = []
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        # Move the data to the GPU
        timeseries = data["historical_y"]
        input_mask = data["input_mask"]
        forecast = data["future_y"]
        with torch.cuda.amp.autocast():
            output = model(x_enc=timeseries, mask=input_mask)
        loss = criterion(output.forecast, forecast)

        accelerator.backward(loss)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

    losses = np.array(losses)
    average_loss = np.average(losses)
    tqdm.write(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

    # Step the learning rate scheduler
    scheduler.step()

    # Evaluate the model on the test split
    if val_dataloader:
        trues, preds, histories, losses = [], [], [], []
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_dataloader, total=len(val_dataloader)):
                # Move the data to the specified device
                timeseries = data["historical_y"]
                input_mask = data["input_mask"]
                forecast = data["future_y"]

                with torch.cuda.amp.autocast():
                    output = model(x_enc=timeseries, mask=input_mask)

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
            f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f}Test MAE: {metrics.mae:.3f}"
        )
    cur_epoch += 1
    return cur_epoch


def _check_device(device):
    if device == "auto":
        return device
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


def _same_index(data: pd.DataFrame):
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(lambda x: x.equals(data.iloc[0])).all(), (
        "All series must has the same index"
    )
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data: pd.DataFrame):
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr


class MomentPytorchDataset(Dataset):
    """Customized Pytorch dataset for the momentfm model."""

    def __init__(self, y, fh, seq_len, device):
        self.y = y
        self.moment_seq_len = 512  # forced seq_len by pre-trained model
        self.seq_len = seq_len
        self.fh = fh
        self.shape = y.shape
        self.device = device

        # multi-index conversion
        if isinstance(y.index, pd.MultiIndex):
            self.y = _frame2numpy(y)
        else:
            self.y = np.expand_dims(y.values, axis=0)

        # n_timestamps should be the seq length for a single series in both
        # cases, multivariate dataframe
        # or a panel/hierarchical dataset
        # if hier/panel, self.n_sequences will be > 1, else will be = 1 if
        # # a regular multivariate df
        self.n_sequences, self.n_timestamps, self.n_columns = self.y.shape

        # self.single_length is defined as the length of one series under
        # one instance in the panel/hier case
        # else it is just the seq_len of the multivariate data
        self.single_length = self.n_timestamps - self.seq_len - self.fh + 1

        # code block to figure out masking sizes in case seq_len < 512
        if self.seq_len < self.moment_seq_len:
            self._pad_shape = (self.moment_seq_len - self.seq_len, self.n_columns)
            self.input_mask = _create_mask(
                self.seq_len, self.moment_seq_len - self.seq_len
            ).float()
            # Concatenate the tensors
        elif self.seq_len > self.moment_seq_len:
            # for now if seq_len > 512 than we reduce it back to 512
            self.seq_len = 512
            self.input_mask = _create_mask(self.seq_len).float()
        else:
            self.input_mask = _create_mask(self.seq_len).float()

    def __len__(self):
        """Return length of dataset."""
        # in the case of a regular multivariate df, we just return the trivial case
        # self.n_timestamps - self.seq_len - self.fh + 1
        # but in case the data is panel/hier, we need to count the total
        # #number of instances
        # i.e self.n_sequences
        return self.single_length * self.n_sequences

    def __getitem__(self, i):
        """Return dataset items from index i."""
        # batches must be returned in format (B, C, S)
        # where B = batch_size, C = channels, S = sequence_length
        from torch import from_numpy

        # select the correct instance
        n = i // self.single_length
        # select the correct timepoint starting index based on the selected instance
        m = i % self.single_length

        hist_end = m + self.seq_len
        pred_end = m + self.seq_len + self.fh

        historical_y = from_numpy(self.y[n, m:hist_end, :]).float()
        # historical_y = historical_y.reshape(historical_y.shape[1], -1)

        if self.seq_len < self.moment_seq_len:
            historical_y = _create_padding(historical_y, self._pad_shape)
        historical_y = historical_y.float().T
        future_y = from_numpy(self.y[n, hist_end:pred_end, :]).float().T
        # future_y = future_y.reshape(future_y.shape[1], -1).T
        return {
            "future_y": future_y,
            "historical_y": historical_y,
            "input_mask": self.input_mask,
        }
