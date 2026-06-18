__all__ = ["ConvTimeNet_backbone"]
__author__ = ["Tanuj-Taneja1"]

import copy

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        return activation()


class SublayerConnection(nn.Module):
    def __init__(self, enable_res_parameter, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            # print(self.a)
            # print(torch.mean(torch.abs(x) / torch.abs(out_x)))
            return x + self.dropout(self.a * out_x)


class _ConvEncoderLayer(nn.Module):
    def __init__(
        self,
        kernel_size,
        d_model,
        d_ff=256,
        dropout=0.1,
        activation="relu",
        enable_res_param=True,
        norm="batch",
        small_ks=3,
        re_param=True,
        device="cpu",
    ):
        super().__init__()

        self.norm_tp = norm
        self.re_param = re_param

        # DeepWise Conv. Add & Norm
        if self.re_param:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(
                d_model,
                d_model,
                self.large_ks,
                stride=1,
                padding="same",
                groups=d_model,
            )
            self.DW_conv_small = nn.Conv1d(
                d_model,
                d_model,
                self.small_ks,
                stride=1,
                padding="same",
                groups=d_model,
            )
            self.DW_infer = nn.Conv1d(
                d_model,
                d_model,
                self.large_ks,
                stride=1,
                padding="same",
                groups=d_model,
            )
        else:
            self.DW_conv = nn.Conv1d(
                d_model, d_model, kernel_size, stride=1, padding="same", groups=d_model
            )

        self.dw_act = get_activation_fn(activation)

        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = (
            nn.BatchNorm1d(d_model) if norm == "batch" else nn.LayerNorm(d_model)
        )

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1, 1),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, 1, 1),
        )

        # Add & Norm
        self.sublayerconnect2 = SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = (
            nn.BatchNorm1d(d_model) if norm == "batch" else nn.LayerNorm(d_model)
        )

    def _get_merge_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad

        module_output = copy.deepcopy(self.DW_conv_large)

        module_output.weight = nn.Parameter(
            module_output.weight
            + nn.functional.pad(
                self.DW_conv_small.weight, (left_pad, right_pad), value=0
            )
        )

        module_output.bias = nn.Parameter(module_output.bias + self.DW_conv_small.bias)

        self.DW_infer = module_output

    def forward(self, src):  # [B, C, L]
        ## Deep-wise Conv Layer
        if not self.re_param:
            src = self.DW_conv(src)
        else:
            if self.training:  # training phase
                large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                src = self.sublayerconnect1(src, self.dw_act(large_out + small_out))
            else:  # testing phase
                self._get_merge_param()
                merge_out = self.DW_infer(src)
                src = self.sublayerconnect1(src, self.dw_act(merge_out))

        src = src.permute(0, 2, 1) if self.norm_tp != "batch" else src
        src = self.dw_norm(src)
        src = src.permute(0, 2, 1) if self.norm_tp != "batch" else src

        ## Position-wise Conv Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm

        src2 = self.sublayerconnect2(
            src, src2
        )  # Add: residual connection with residual dropout

        # Norm: batchnorm or layernorm
        src2 = src2.permute(0, 2, 1) if self.norm_tp != "batch" else src2
        src2 = self.norm_ffn(src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != "batch" else src2

        return src2


class _ConvEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        kernel_size=[19, 19, 29, 29, 37, 37],
        dropout=0.1,
        activation="gelu",
        n_layers=3,
        enable_res_param=False,
        norm="batch",
        re_param=False,
        device="cpu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ConvEncoderLayer(
                    kernel_size[i],
                    d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    enable_res_param=enable_res_param,
                    norm=norm,
                    re_param=re_param,
                    device=device,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class ConvTimeNet_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        d_ff: int = 256,
        dropout=0.1,
        act: str = "relu",
        pooling_tp="max",
        fc_dropout: float = 0.0,
        enable_res_param=False,
        dw_ks=[7, 13, 19],
        norm="batch",
        use_embed=True,
        re_param=False,
        device: str = "cpu",
    ):
        """ConvTST is a Transformer that takes continuous time series as inputs.

        As mentioned in the paper, the input must be standardized by_var based on
        the entire training set.
        Args:
        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels)
            x seq_len (aka time steps)
        """
        super().__init__()
        assert n_layers == len(dw_ks), "dw_ks should match the n_layers!"

        self.c_out, self.seq_len = c_out, seq_len

        # Input Embedding
        self.use_embed = use_embed
        self.W_P = nn.Linear(c_in, d_model)

        # Positional encoding
        # W_pos = torch.empty((seq_len, d_model), device=device)
        # nn.init.uniform_(W_pos, -0.02, 0.02)
        # self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _ConvEncoder(
            d_model,
            d_ff,
            kernel_size=dw_ks,
            dropout=dropout,
            activation=act,
            n_layers=n_layers,
            enable_res_param=enable_res_param,
            norm=norm,
            re_param=re_param,
            device=device,
        )

        self.flatten = nn.Flatten()

        # Head
        self.head_nf = seq_len * d_model if pooling_tp == "cat" else d_model
        self.head = self.create_head(
            self.head_nf, c_out, act=act, pooling_tp=pooling_tp, fc_dropout=fc_dropout
        )

    def create_head(
        self, nf, c_out, act="gelu", pooling_tp="max", fc_dropout=0.0, **kwargs
    ):
        layers = []
        if pooling_tp == "cat":
            layers = [get_activation_fn(act), self.flatten]
            if fc_dropout:
                layers += [nn.Dropout(fc_dropout)]
        elif pooling_tp == "mean":
            layers = [nn.AdaptiveAvgPool1d(1), self.flatten]
        elif pooling_tp == "max":
            layers = [nn.AdaptiveMaxPool1d(1), self.flatten]

        layers += [nn.Linear(nf, c_out)]

        # could just be used in classifying task
        return nn.Sequential(*layers)

    def forward(self, x):  # x: [bs x nvars x q_len]
        # Input encoding
        u = x
        if self.use_embed:
            u = self.W_P(x.transpose(2, 1))

        # Positional encoding
        # u = self.dropout(u + self.W_pos[:u.shape[1]])   # u: [bs x q_len x d_model]

        # Encoder
        z = self.encoder(u.transpose(2, 1).contiguous())  # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.head(z)  # output: [bs x c_out]
