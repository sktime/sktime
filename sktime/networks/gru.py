"""Gated Recurrent Unit (GRU) for time series classification."""

__author__ = ["fnhirwa"]

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

    NNModule = nn.Module
else:

    class NNModule:
        """Dummy class if torch is unavailable."""


class GRU:
    """Gated Recurrent Unit (GRU) for time series classification.

    Network originally defined in [1]_.

    Parameters
    ----------
    input_size : int
        Number of features in the input time series.
    hidden_dim : int
        Number of features in the hidden state.
    n_layers : int
        Number of recurrent layers.
    batch_first : bool
        If True, then the input and output tensors are provided
        as (batch, seq, feature), default is False.
    bias : bool
        If False, then the layer does not use bias weights, default is True.
    num_classes : int
        Number of classes to predict.
    init_weights : bool
        If True, then the weights are initialized, default is True.
    dropout : float
        Dropout rate to apply. default is 0.0
    fc_dropout : float
        Dropout rate to apply to the fully connected layer. default is 0.0
    bidirectional : bool
        If True, then the GRU is bidirectional, default is False.

    References
    ----------

    .. [1] Cho, Kyunghyun, et al. "Learning phrase representations
        using RNN encoder-decoder for statistical machine translation."
        arXiv preprint arXiv:1406.1078 (2014).
    """

    _tags = {
        # packaging info
        # --------------
        "author": ["fnhirwa"],
        "maintainers": ["fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
    }

    class _GRU_Network(NNModule):
        def __init__(
            self,
            input_size,
            hidden_dim,
            n_layers,
            batch_first,
            bias,
            num_classes,
            init_weights,
            dropout,
            bidirectional,
            fc_dropout,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.init_weights = init_weights
            self.dropout = dropout
            self.bidirectional = bidirectional
            self.gru = nn.GRU(
                input_size,
                hidden_dim,
                num_layers=n_layers,
                batch_first=batch_first,
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.fc_dropout = (
                nn.Dropout(fc_dropout) if self.fc_dropout else nn.Identity()
            )  # noqa: E501
            self.fc = nn.Linear(hidden_dim * (1 + bidirectional), num_classes)
            if init_weights:
                self.apply(self._init_weights)

        def _init_weights(self, module):
            # adapted from https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization #noqa: E501
            # Tensoflow like initialization
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(0)
                    # set forget bias to 1
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size : 2 * hidden_size].fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)

        def forward(self, x):
            """Forward pass through the network.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.
            """
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()

            x = x.transpose(2, 1) if x.dim() == 3 else x
            gru_out, _ = self.gru(x)
            output = gru_out[:, -1, :]
            output = self.fc(self.fc_dropout(output))
            return output

    def __init__(
        self,
        input_size,
        hidden_dim,
        n_layers,
        batch_first=False,
        bias=True,
        num_classes=2,
        init_weights=True,
        dropout=0.0,
        fc_dropout=0.0,
        bidirectional=False,
    ) -> None:
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bias = bias
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.bidirectional = bidirectional

    def build(self):
        """Build the network."""
        return self._GRU_Network(
            self.input_size,
            self.hidden_dim,
            self.n_layers,
            self.batch_first,
            self.bias,
            self.num_classes,
            self.init_weights,
            self.dropout,
            self.bidirectional,
            self.fc_dropout,
        )
