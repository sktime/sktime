"""Simple RNN network for direct multi-step forecasting."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


class RNNForecastNetwork:
    """RNN network builder for direct multi-step time series forecasting.

    Parameters
    ----------
    seq_len : int
        Length of input sequence.
    pred_len : int
        Forecast horizon length.
    hidden_dim : int, default=64
        Hidden state size of the RNN.
    n_layers : int, default=1
        Number of stacked RNN layers.
    dropout : float, default=0.0
        Dropout used between RNN layers when ``n_layers > 1``.
    nonlinearity : {"tanh", "relu"}, default="tanh"
        Non-linearity used by ``nn.RNN``.
    bias : bool, default=True
        Whether to use bias in ``nn.RNN`` and ``nn.Linear``.
    """

    _tags = {
        "authors": ["fkiraly"],
        "maintainers": ["fkiraly"],
        "python_dependencies": ["torch"],
    }

    class _RNNModule(nn_module):
        """Internal PyTorch module for RNN forecasting."""

        def __init__(
            self,
            seq_len,
            pred_len,
            hidden_dim=6,
            n_layers=1,
            dropout=0.0,
            nonlinearity="tanh",
            bias=True,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            self.rnn = nn.RNN(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                nonlinearity=nonlinearity,
                bias=bias,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.proj = nn.Linear(hidden_dim, pred_len, bias=bias)

        def forward(self, x):
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input of shape ``[batch, seq_len, n_vars]``.

            Returns
            -------
            torch.Tensor
                Forecast of shape ``[batch, pred_len, n_vars]``.
            """
            batch_size, _, n_vars = x.shape

            # Process each variable as a separate univariate sequence.
            x = x.permute(0, 2, 1).reshape(batch_size * n_vars, self.seq_len, 1)
            _, h_n = self.rnn(x)
            last_hidden = h_n[-1]

            out = self.proj(last_hidden)
            out = out.reshape(batch_size, n_vars, self.pred_len)
            return out.permute(0, 2, 1)

    def __init__(
        self,
        seq_len,
        pred_len,
        hidden_dim=64,
        n_layers=1,
        dropout=0.0,
        nonlinearity="tanh",
        bias=True,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.nonlinearity = nonlinearity
        self.bias = bias

    def _build(self):
        """Build and return the PyTorch network module."""
        return self._RNNModule(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            nonlinearity=self.nonlinearity,
            bias=self.bias,
        )
