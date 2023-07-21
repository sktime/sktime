"""Deep Learning Forecasters using LTSF-Linear Models."""

import torch

from sktime.forecasting.deep_learning.base import BaseDeepNetworkPyTorch


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

    # TODO: fix docstring

    def __init__(
        self,
        seq_len=10,  # L : Historical data
        pred_len=1,  # T : Future predictions
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.003,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
    ):
        from sktime.networks.ltsf import LTSFLinearNetwork

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.individual = individual

        super().__init__(
            network=LTSFLinearNetwork(
                seq_len,
                pred_len,
                in_channels,
                individual,
            ),
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
