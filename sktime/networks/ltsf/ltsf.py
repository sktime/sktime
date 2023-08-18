"""Deep Learning Forecasters using LTSF-Linear Models."""

import torch
import torch.nn as nn

from sktime.networks.base import BaseDeepNetworkPyTorch


class LTSFLinearNetwork(nn.Module):
    """LSTF-Linear Network.

    Just one Linear layer. Taken from _[1].

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    in_channels : int, default=None
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."

    References
    ----------
    [1] @inproceedings{Zeng2022AreTE,
        title={Are Transformers Effective for Time Series Forecasting?},
        author={Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
        journal={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2023}
    }
        [Source]: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        in_channels=1,
        individual=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.individual = individual

        if self.individual:
            self.Linear = nn.ModuleList()
            for _ in range(self.in_channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        """Forward pass for LSTF-Linear Network.

        Parameters
        ----------
        x : torch.Tensor
            torch.Tensor of shape [Batch, Input Sequence Length, Channel]

        Returns
        -------
        x : torch.Tensor
            output of Linear Model. x.shape = [Batch, Output Length, Channel]
        """
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
            ).to(x.device)
            for i in range(self.in_channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [Batch, Output Length, Channel]


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

    _tags = {
        "scitype:y": "both",
        "y_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
    }

    def __init__(
        self,
        seq_len,  # L : Historical data
        pred_len,  # T : Future predictions
        target=None,
        features=None,
        scale=True,
        in_channels=1,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        num_epochs=16,
        batch_size=8,
        individual=False,
        shuffle=True,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target = target
        self.features = features
        self.scale = scale
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.individual = individual
        self.shuffle = shuffle
        self.network = LTSFLinearNetwork(
            seq_len,
            pred_len,
            in_channels,
            individual,
        )

        super().__init__()

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
        params : dict or list of dict
        """
        params = []

        return params
