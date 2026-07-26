"""ConvTimeNet (PyTorch) network components for time series classification."""

__author__ = ["Tanuj-Taneja1"]

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")

from sktime.networks.convtimenet._convtimenet_backbone import (
    ConvTimeNet_backbone,
)
from sktime.networks.convtimenet._dlutils import DeformablePatch


class ConvTimeNet:
    _tags = {
        "authors": ["Tanuj-Taneja1"],
        "maintainers": ["Tanuj-Taneja1"],
        "python_dependencies": ["torch"],
    }

    class _ConvTimeNet(nn.Module):
        def __init__(
            self,
            enc_in: int,
            d_model: int,
            seq_len: int,
            patch_size: int,
            patch_stride: int,
            n_classes: int,
            dropout: float,
            d_ff: int,
            dw_ks: list[int],
            device: str = "cpu",
        ):
            super().__init__()

            #  DePatch Embedding
            self.depatchEmbedding = DeformablePatch(
                seq_len=seq_len,
                patch_size=patch_size,
                stride=patch_stride,
                in_feats=enc_in,
                out_feats=d_model,
            )

            # 🧠 ConvTimeNet Backbone
            new_len = self.depatchEmbedding.new_len
            c_in = d_model
            c_out = n_classes
            block_num = len(dw_ks)
            enable_res_param = True
            re_param = True

            self.main_net = ConvTimeNet_backbone(
                c_in=c_in,
                c_out=c_out,
                seq_len=new_len,
                n_layers=block_num,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                act="gelu",
                pooling_tp="max",
                fc_dropout=0.0,
                enable_res_param=enable_res_param,
                dw_ks=dw_ks,
                norm="batch",
                use_embed=False,
                re_param=re_param,
                device=device,
            )

        def forward(
            self,
            X=None,
            x_enc=None,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None,
            mask=None,
        ):
            x = X if X is not None else x_enc
            out_patch = self.depatchEmbedding(x)  # [bs, features]
            output = self.main_net(out_patch.permute(0, 2, 1))
            return output

    # ---------------- Outer wrapper ----------------
    def __init__(
        self,
        enc_in: int,
        d_model: int,
        seq_len: int,
        patch_size: int,
        patch_stride: int,
        n_classes: int,
        dropout: float,
        d_ff: int,
        dw_ks: list[int],
        device: str = "cpu",
    ):
        # Store params for later use
        self.params = dict(
            enc_in=enc_in,
            d_model=d_model,
            seq_len=seq_len,
            patch_size=patch_size,
            patch_stride=patch_stride,
            n_classes=n_classes,
            dropout=dropout,
            d_ff=d_ff,
            dw_ks=dw_ks,
            device=device,
        )
        self.model = None

    def build(self):
        # Actually construct the nn.Module
        self.model = ConvTimeNet._ConvTimeNet(**self.params)
        return self.model
