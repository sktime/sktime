"""Gated Recurrent Unit (GRU) network in PyTorch."""

__authors__ = ["fnhirwa"]
__all__ = ["GRUNetworkTorch"]

from collections.abc import Callable

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class GRUNetworkTorch(NNModule):
    """Gated Recurrent Unit (GRU) network for time series classification.

    Network originally defined in [1]_, [2]_ and [3]_.

    Parameters
    ----------
    input_size : int
        Number of features in the input time series.
    num_classes : int
        Number of outputs, i.e., the number of classes. Use ``1`` for regression.
    hidden_dim : int, default=256
        Number of features in the hidden state.
    n_layers : int, default=4
        Number of recurrent layers.
    batch_first : bool, default=False
        If True, then the input tensor is provided as (batch, seq, feature),
        otherwise as (seq, batch, feature).
    bias : bool, default=True
        If False, then the layer does not use bias weights.
    init_weights : bool, default=True
        If True, then the weights are initialized with a TensorFlow like
        initialization.
    dropout : float, default=0.0
        Dropout rate applied between the recurrent layers.
    fc_dropout : float, default=0.0
        Dropout rate applied before the fully connected layer.
    bidirectional : bool, default=False
        If True, then the GRU is bidirectional.
    activation : Callable or None, default=None
        Activation applied to the output layer. If None, the network returns
        raw outputs, i.e., logits.

    References
    ----------
    .. [1] Cho, Kyunghyun, et al. "Learning phrase representations
        using RNN encoder-decoder for statistical machine translation."
        arXiv preprint arXiv:1406.1078 (2014).
    .. [2] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio.
        Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
        arXiv preprint arXiv:1412.3555 (2014).
    .. [3] https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fnhirwa"],
        "maintainers": ["fnhirwa", "srupat"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "GRUNetworkTorch",
        input_size: int,
        num_classes: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        batch_first: bool = False,
        bias: bool = True,
        init_weights: bool = True,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        bidirectional: bool = False,
        activation: Callable | None = None,
    ):
        super().__init__()

        nnGRU = _safe_import("torch.nn.GRU")
        nnDropout = _safe_import("torch.nn.Dropout")
        nnIdentity = _safe_import("torch.nn.Identity")
        nnLinear = _safe_import("torch.nn.Linear")

        self._import_cache = {}

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.bidirectional = bidirectional
        self.init_weights = init_weights

        self.gru = nnGRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=batch_first,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.out_dropout = nnDropout(fc_dropout) if fc_dropout else nnIdentity()
        self.fc = nnLinear(hidden_dim * (1 + bidirectional), num_classes)
        self.activation = activation

        if self.init_weights:
            self.apply(self._init_weights)

    def _torch_op(self, import_path):
        """Lazily import and cache a torch callable used in the forward pass."""
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]

    def _init_weights(self, module):
        """Apply a TensorFlow like initialization to the recurrent weights.

        Adapted from
        https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization

        Parameters
        ----------
        module : torch.nn.Module
            Module whose parameters are initialized in place.
        """
        xavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")
        orthogonal_ = _safe_import("torch.nn.init.orthogonal_")

        for name, param in module.named_parameters():
            if "weight_ih" in name:
                xavier_uniform_(param.data)
            elif "weight_hh" in name:
                orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor containing the time series data, of shape
            ``(batch_size, seq_len, input_size)`` if ``batch_first``, else
            ``(seq_len, batch_size, input_size)``.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes)
            Class scores. Raw outputs, i.e., logits, unless ``activation``
            is passed.
        """
        if isinstance(X, np.ndarray):
            from_numpy = self._torch_op("torch.from_numpy")
            X = from_numpy(X).float()

        X, _ = self.gru(X)
        # keep the hidden state of the last time step; the time axis is 1 for
        # batch-first inputs and 0 otherwise
        X = X[:, -1, :] if self.batch_first else X[-1, :, :]
        X = self.out_dropout(X)
        X = self.fc(X)
        if self.activation is not None:
            X = self.activation(X)
        return X
