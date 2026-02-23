"""MLP (Multi-Layer Perceptron) network for Time Series Forecasting."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


class MLPForecastNetwork:
    """MLP network builder for time series forecasting.

    Based on the M4 competition MLP benchmark model [1]_.

    Parameters
    ----------
    seq_len : int
        Length of input sequence.
    pred_len : int
        Length of prediction (forecast horizon).
    hidden_dim : int, default=100
        Number of units in each hidden layer.
    n_layers : int, default=3
        Number of hidden layers.
    dropout : float, default=0.0
        Dropout rate between hidden layers.

    References
    ----------
    .. [1] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. 2018.
    The M4 Competition: Results, findings, conclusion and way forward.
    International Journal of Forecasting.
    https://github.com/Mcompetitions/M4-methods
    """

    _tags = {
        "authors": ["fkiraly"],
        "maintainers": ["fkiraly"],
        "python_dependencies": ["torch"],
    }

    class _MLPModule(nn_module):
        """Internal PyTorch module for MLP network."""

        def __init__(self, seq_len, pred_len, hidden_dim, n_layers, dropout):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len

            layers = []
            in_features = seq_len

            # Hidden layers
            for _ in range(n_layers):
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_features = hidden_dim

            # Output layer
            layers.append(nn.Linear(hidden_dim, pred_len))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape [batch, seq_len, n_vars]

            Returns
            -------
            torch.Tensor
                Output tensor of shape [batch, pred_len, n_vars]
            """
            # Input: [batch, seq_len, n_vars]
            batch_size, _, n_vars = x.shape

            # Permute to [batch, n_vars, seq_len]
            x = x.permute(0, 2, 1)

            # Reshape to [batch * n_vars, seq_len]
            x = x.reshape(batch_size * n_vars, -1)

            # Forward through network
            out = self.network(x)  # [batch * n_vars, pred_len]

            # Reshape back to [batch, n_vars, pred_len]
            out = out.reshape(batch_size, n_vars, -1)

            # Permute to [batch, pred_len, n_vars]
            return out.permute(0, 2, 1)

    def __init__(self, seq_len, pred_len, hidden_dim=100, n_layers=3, dropout=0.0):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

    def _build(self):
        """Build and return the PyTorch network module."""
        return self._MLPModule(
            self.seq_len,
            self.pred_len,
            self.hidden_dim,
            self.n_layers,
            self.dropout,
        )
