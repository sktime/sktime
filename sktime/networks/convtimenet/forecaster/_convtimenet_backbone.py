__all__ = ["ConvTimeNet_backbone"]

# Cell
import copy

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")
F = _safe_import("torch.nn.functional")
tensor = _safe_import("torch.Tensor")


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


# Cell
class ConvTimeNet_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        n_layers: int = 6,
        dw_ks=[9, 11, 15, 21, 29, 39],
        d_model=64,
        d_ff: int = 256,
        norm: str = "batch",
        dropout: float = 0.0,
        act: str = "gelu",
        head_dropout=0,
        padding_patch=None,
        head_type="flatten",
        revin=True,
        affine=True,
        subtract_last=False,
        deformable=True,
        enable_res_param=True,
        re_param=True,
        re_param_kernel=3,
        device="cuda:0",
    ):
        from ._patch_layers import DepatchSampling
        from ._revin import RevIN

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching / Deformable Patching
        self.deformable = deformable
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        seq_len = (patch_num - 1) * self.stride + self.patch_len
        if deformable:
            self.deformable_sampling = DepatchSampling(
                c_in, seq_len, self.patch_len, self.stride, device
            )

        # Backbone
        self.backbone = ConviEncoder(
            patch_num=patch_num,
            patch_len=patch_len,
            kernel_size=dw_ks,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            norm=norm,
            dropout=dropout,
            act=act,
            enable_res_param=enable_res_param,
            re_param=re_param,
            re_param_kernel=re_param_kernel,
            device=device,
        )

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "flatten":
            self.head = Flatten_Head(
                self.n_vars, self.head_nf, target_window, head_dropout=head_dropout
            )
        else:
            raise ValueError(f"No such head@ {self.head}")

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)

        if not self.deformable:
            z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            z = self.deformable_sampling(z)

        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.n_vars = n_vars

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ConviEncoder(nn.Module):  # i means channel-independent
    def __init__(
        self,
        patch_num,
        patch_len,
        kernel_size=[11, 15, 21, 29, 39, 51],
        n_layers=6,
        d_model=128,
        d_ff=256,
        norm="batch",
        dropout=0.0,
        act="gelu",
        enable_res_param=True,
        re_param=True,
        re_param_kernel=3,
        device="cuda:0",
    ):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input embedding
        self.W_P = nn.Linear(patch_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = ConvEncoder(
            kernel_size,
            d_model,
            d_ff=d_ff,
            norm=norm,
            dropout=dropout,
            activation=act,
            enable_res_param=enable_res_param,
            n_layers=n_layers,
            re_param=re_param,
            re_param_kernel=re_param_kernel,
            device=device,
        )

    def forward(self, x):  # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # u: [bs * nvars x patch_num x d_model]
        # u = self.dropout(u + self.W_pos)
        # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z


class ConvEncoder(nn.Module):
    def __init__(
        self,
        kernel_size,
        d_model,
        d_ff=None,
        norm="batch",
        dropout=0.0,
        activation="gelu",
        enable_res_param=True,
        n_layers=3,
        re_param=True,
        re_param_kernel=3,
        device="cuda:0",
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ConvEncoderLayer(
                    d_model,
                    d_ff=d_ff,
                    kernel_size=kernel_size[i],
                    dropout=dropout,
                    activation=activation,
                    enable_res_param=enable_res_param,
                    norm=norm,
                    re_param=re_param,
                    small_ks=re_param_kernel,
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


class SublayerConnection(nn.Module):
    def __init__(self, enable_res_parameter, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x)  #
        else:
            return x + self.dropout(self.a * out_x)


class ConvEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int = 256,
        kernel_size: int = 9,
        dropout: float = 0.1,
        activation: str = "gelu",
        enable_res_param=True,
        norm="batch",
        re_param=True,
        small_ks=3,
        device="cuda:0",
    ):
        super().__init__()

        self.norm_tp = norm
        self.re_param = re_param

        if not re_param:
            self.DW_conv = nn.Conv1d(
                d_model, d_model, kernel_size, 1, "same", groups=d_model
            )
        else:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(
                d_model, d_model, kernel_size, stride=1, padding="same", groups=d_model
            )
            self.DW_conv_small = nn.Conv1d(
                d_model, d_model, small_ks, stride=1, padding="same", groups=d_model
            )
            self.DW_infer = nn.Conv1d(
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

    def _get_merged_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        module_output = copy.deepcopy(self.DW_conv_large)
        module_output.weight = torch.nn.Parameter(
            module_output.weight
            + F.pad(self.DW_conv_small.weight, (left_pad, right_pad), value=0)
        )
        # module_output.bias += self.DW_conv_small.bias
        module_output.bias = torch.nn.Parameter(
            module_output.bias + self.DW_conv_small.bias
        )
        self.DW_infer = module_output

    def forward(self, src: torch.Tensor) -> torch.Tensor:  # [B, C, L]
        ## Deep-wise Conv Layer
        if not self.re_param:
            out_x = self.DW_conv(src)
        else:
            if self.training:  # training phase
                large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                out_x = large_out + small_out
            else:  # testing phase
                self._get_merged_param()
                out_x = self.DW_infer(src)

        src2 = self.dw_act(out_x)
        # print(src2.shape); exit(0)

        src = self.sublayerconnect1(src, src2)
        src = src.permute(0, 2, 1) if self.norm_tp != "batch" else src
        src = self.dw_norm(src)
        src = src.permute(0, 2, 1) if self.norm_tp != "batch" else src

        ## Position-wise Conv Feed-Forward
        src2 = self.ff(src)
        src2 = self.sublayerconnect2(
            src, src2
        )  # Add: residual connection with residual dropout

        # Norm: batchnorm or layernorm
        src2 = src2.permute(0, 2, 1) if self.norm_tp != "batch" else src2
        src2 = self.norm_ffn(src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != "batch" else src2

        return src2
