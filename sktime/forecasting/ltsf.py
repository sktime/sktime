"""Deep Learning Forecasters using LTSF-Linear Models."""

import torch

from sktime.networks.base import BaseDeepNetworkPyTorch


class LTSFLinearForecaster(BaseDeepNetworkPyTorch):
    """LTSF-Linear Forecaster.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    lr : float
        learning rate
    num_epochs : int
        number of epochs to train
    batch_size : int
        number of training examples per batch
    in_channels : int, default=None
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."
    optimizer : torch.optim.Optimizer, default=torch.optim.Adam
        optimizer to be used for training
    criterion : torch.nn Loss Function, default=torch.nn.MSELoss
        loss function to be used for training
    """

    def __init__(
        self,
        seq_len,  # L : Historical data
        pred_len,  # T : Future predictions
        lr,
        num_epochs,
        batch_size,
        shuffle=False,
        in_channels=1,
        individual=False,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss,
    ):
        from sktime.networks.ltsf import LTSFLinearNetwork

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.individual = individual

        self._network = LTSFLinearNetwork(
            seq_len,
            pred_len,
            in_channels,
            individual,
        )

        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.criterion = criterion

        super().__init__()
