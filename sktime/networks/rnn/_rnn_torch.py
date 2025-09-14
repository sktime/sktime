"""Recurrent Neural Network (RNN) for Classification and Regression in PyTorch."""

__author__ = ["RecreationalMath"]

from sktime.utils.dependencies import _safe_import

nn = _safe_import("torch.nn")
torch = _safe_import("torch")


class RNNNetworkTorch(nn.Module):
    """Establish the network structure for an RNN in PyTorch.

    Parameters
    ----------
    input_size     : int
        The number of expected features in the input x
    hidden_size    : int, default = 6
        The number of features in the hidden state h
    num_layers     : int, default = 1
        Number of recurrent layers.
        E.g., setting num_layers=2 would mean stacking two RNNs together to form
        a stacked RNN, with the second RNN taking in outputs of the first RNN
        and computing the final results.
    random_state   : int, default = 0
        seed to any needed random actions
    """

    def __init__(
        self,
        input_size,
        hidden_size=6,
        num_layers=1,
        random_state=0,
    ):
        if nn is None:
            raise ImportError(
                "RNNNetworkTorch requires torch to be installed. "
                "Please install torch or use a different estimator."
            )
        self.random_state = random_state
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_length, input_size)
            Input tensor containing the time series data.

        Returns
        -------
        out : torch.Tensor of shape (batch_size, seq_length, hidden_size)
            Output tensor containing the hidden states for each time step.
        """
        h0 = nn.init.xavier_uniform_(
            torch.empty(self.num_layers, x.size(0), self.hidden_size)
        )
        out, _ = self.rnn(x, h0)
        return out
