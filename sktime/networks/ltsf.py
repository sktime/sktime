"""LSTF-Linear Models."""
import torch
import torch.nn as nn


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
