# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""iTransformer Network."""

__author__ = ["TenFinges"]

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


class ITransformerNetwork:
    """iTransformer Network.

    Implements the iTransformer architecture proposed in
    "iTransformer: Inverted Transformers are Effective for Time Series Forecasting".
    Instead of embedding time points, it embeds the entire time series (variate)
    as a token.

    Parameters
    ----------
    seq_len : int
        Input sequence length (lookback window).
    pred_len : int
        Output sequence length (forecast horizon).
    num_variates : int
        Number of input variates (channels).
    d_model : int, default=512
        Dimension of the transformer hidden state.
    nhead : int, default=8
        Number of attention heads.
    num_encoder_layers : int, default=2
        Number of transformer encoder layers.
    dim_feedforward : int, default=2048
        Dimension of the feedforward network model.
    dropout : float, default=0.1
        Dropout value.
    activation : str, default='relu'
        Activation function of the transformer encoder.
    """

    _tags = {
        "authors": ["TenFinges"],
        "maintainers": ["TenFinges"],
        "python_dependencies": "torch",
    }

    class _ITransformerNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            pred_len,
            num_variates,
            d_model=512,
            nhead=8,
            num_encoder_layers=2,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        ):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.num_variates = num_variates

            # 1. Embedding: Embed the entire series (seq_len) into d_model
            # Project T -> d_model
            # Note: We use a Linear layer to project the time dimension to
            # feature dimension
            self.embedding = nn.Linear(seq_len, d_model)

            # 2. Encoder: Learn correlations among variates
            # batch_first=True ensures input format is (Batch, Seq, Feature)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

            # 3. Projection: Project d_model -> pred_len
            self.projection = nn.Linear(d_model, pred_len)

        def forward(self, x):
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (Batch, seq_len, num_variates)

            Returns
            -------
            torch.Tensor
                Output tensor of shape (Batch, pred_len, num_variates)
            """
            # x: [Batch, T, N]

            # Invert: [Batch, T, N] -> [Batch, N, T]
            # In iTransformer, each variate is a token.
            x = x.permute(0, 2, 1)

            # Embedding: [Batch, N, T] -> [Batch, N, d_model]
            x = self.embedding(x)

            # Encoder: [Batch, N, d_model] -> [Batch, N, d_model]
            # Attention is calculated between variates (N tokens)
            x = self.encoder(x)

            # Projection: [Batch, N, d_model] -> [Batch, N, pred_len]
            x = self.projection(x)

            # Invert back: [Batch, N, pred_len] -> [Batch, pred_len, N]
            x = x.permute(0, 2, 1)

            return x

    def __init__(
        self,
        seq_len,
        pred_len,
        num_variates,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_variates = num_variates
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

    def _build(self):
        return self._ITransformerNetwork(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            num_variates=self.num_variates,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )
