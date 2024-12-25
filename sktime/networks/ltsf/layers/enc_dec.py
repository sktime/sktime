"""Extra LTSF Model Layers."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


class LTSFTransformerConvLayer:
    """LTSFTransformerConvLayer."""

    def __init__(self, in_channels):
        self.in_channels = in_channels

    def _build(self):
        return self._LTSFTransformerConvLayer(self.in_channels)

    class _LTSFTransformerConvLayer(nn_module):
        def __init__(self, in_channels):
            super().__init__()
            self.downConv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=2,
                padding_mode="circular",
            )
            self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.ELU()
            self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.downConv(x.permute(0, 2, 1))
            x = self.norm(x)
            x = self.activation(x)
            x = self.maxPool(x)
            x = x.transpose(1, 2)
            return x


class LTSFTransformerEncoderLayer:
    """LTSFTransformerEncoderLayer."""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        self.attention = attention
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

    def _build(self):
        return self._LTSFTransformerEncoderLayer(
            self.attention, self.d_model, self.d_ff, self.dropout, self.activation
        )

    class _LTSFTransformerEncoderLayer(nn_module):
        def __init__(
            self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"
        ):
            from torch.nn.functional import gelu, relu

            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.attention = attention
            self.conv1 = nn.Conv1d(
                in_channels=d_model, out_channels=d_ff, kernel_size=1
            )
            self.conv2 = nn.Conv1d(
                in_channels=d_ff, out_channels=d_model, kernel_size=1
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = relu if activation == "relu" else gelu

        def forward(self, x, attn_mask=None):
            new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
            x = x + self.dropout(new_x)

            y = x = self.norm1(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

            return self.norm2(x + y), attn


class LTSFTransformerEncoder:
    """LTSFTransformerEncoder."""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers
        self.norm_layer = norm_layer

    def _build(self):
        return self._LTSFTransformerEncoder(
            self.attn_layers, self.conv_layers, self.norm_layer
        )

    class _LTSFTransformerEncoder(nn_module):
        def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
            super().__init__()
            self.attn_layers = nn.ModuleList(attn_layers)
            self.conv_layers = (
                nn.ModuleList(conv_layers) if conv_layers is not None else None
            )
            self.norm = norm_layer

        def forward(self, x, attn_mask=None):
            # x [B, L, D]
            attns = []
            if self.conv_layers is not None:
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    x = conv_layer(x)
                    attns.append(attn)
                x, attn = self.attn_layers[-1](x)
                attns.append(attn)
            else:
                for attn_layer in self.attn_layers:
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    attns.append(attn)

            if self.norm is not None:
                x = self.norm(x)

            return x, attns


class LTSFTransformerDecoderLayer:
    """LTSFTransformerDecoderLayer."""

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

    def _build(self):
        return self._LTSFTransformerDecoderLayer(
            self.self_attention,
            self.cross_attention,
            self.d_model,
            self.d_ff,
            self.dropout,
            self.activation,
        )

    class _LTSFTransformerDecoderLayer(nn_module):
        def __init__(
            self,
            self_attention,
            cross_attention,
            d_model,
            d_ff=None,
            dropout=0.1,
            activation="relu",
        ):
            from torch.nn.functional import gelu, relu

            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.self_attention = self_attention
            self.cross_attention = cross_attention
            self.conv1 = nn.Conv1d(
                in_channels=d_model, out_channels=d_ff, kernel_size=1
            )
            self.conv2 = nn.Conv1d(
                in_channels=d_ff, out_channels=d_model, kernel_size=1
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = relu if activation == "relu" else gelu

        def forward(self, x, cross, x_mask=None, cross_mask=None):
            x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
            x = self.norm1(x)

            x = x + self.dropout(
                self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
            )

            y = x = self.norm2(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

            return self.norm3(x + y)


class LTSFTransformerDecoder:
    """LTSFTransformerDecoder."""

    def __init__(self, layers, norm_layer=None, projection=None):
        self.layers = layers
        self.norm_layer = norm_layer
        self.projection = projection

    def _build(self):
        return self._LTSFTransformerDecoder(
            self.layers, self.norm_layer, self.projection
        )

    class _LTSFTransformerDecoder(nn_module):
        def __init__(self, layers, norm_layer=None, projection=None):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.norm = norm_layer
            self.projection = projection

        def forward(self, x, cross, x_mask=None, cross_mask=None):
            for layer in self.layers:
                x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

            if self.norm is not None:
                x = self.norm(x)

            if self.projection is not None:
                x = self.projection(x)
            return x
