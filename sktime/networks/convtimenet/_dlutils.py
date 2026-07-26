__author__ = ["Tanuj-Taneja1"]

import math

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


def weights_init(mod):
    """
    Initialize weights of netG, netD, and netE.

    :param m: Module whose weights need initialization.
    :return: None
    """
    classname = mod.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find("BatchNorm") != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(mod.weight)
        if mod.bias is not None:
            mod.bias.data.fill_(0.01)


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        return activation()


class PositionalEncoding(nn.Module):  # static PE
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, seqlen=10, pos=0):  # (bs, seq_len, feats)
        idx = 0 if x.shape[0] == seqlen else 1
        x = x + self.pe[pos : pos + x.size(idx), :]
        return self.dropout(x)


class SimplePatch(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len,
        patch_size,
        stride=1,
        norm="batch",
        padding_tp=None,
        device="cpu",
    ):
        super().__init__()

        self.ptw = patch_size

        if padding_tp == "same":
            stride = 1
            self.l_pad = 1 * (patch_size - 1) // 2
            self.r_pad = 1 * (patch_size - 1) - self.l_pad
            n_padding = self.l_pad + self.r_pad

        else:
            n_stride = (seq_len - self.ptw) // stride + 1
            n_padding = n_stride * stride + self.ptw - seq_len
            self.l_pad = n_padding // 2
            self.r_pad = n_padding - self.l_pad

        self.new_len = seq_len + n_padding
        self.patch_net = (
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.ptw,
                    stride=stride,
                    padding=0,
                ),
            )
            .to(device)
            .apply(weights_init)
        )

        self.norm_tp = norm
        self.norm = (
            nn.LayerNorm(out_channels)
            if norm == "layer"
            else nn.BatchNorm1d(out_channels)
        )

    def forward(self, X):  # Input: (bs, window, channel)
        X = X.permute(0, 2, 1)
        X = nn.functional.pad(X, (self.l_pad, self.r_pad), mode="constant", value=0)
        X = self.patch_net(X)

        if self.norm_tp == "layer":
            X = self.norm(X.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            X = self.norm(X)
        return X


class BoxCoder(nn.Module):
    def __init__(
        self,
        patch_count,
        patch_stride,
        patch_size,
        seq_len,
        channels,
        weights=(1.0, 1.0),
        tanh=False,
        device="cpu",
    ):
        super().__init__()

        self.tanh = tanh
        self.seq_len = seq_len
        self.channels = channels
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.patch_stride = patch_stride

        self._generate_anchor(device=device)
        self.weights = weights

    # compute the center points
    def _generate_anchor(self, device="cpu"):
        anchors = []
        self.S_bias = (self.patch_size - 1) / 2

        for i in range(self.patch_count):
            x = i * self.patch_stride + 0.5 * (self.patch_size - 1)
            anchors.append(x)
        anchors = torch.as_tensor(anchors, device=device)
        self.register_buffer("anchor", anchors)

    def forward(self, boxes):
        self.bound = self.decode(boxes)  # (bs, patch_count, 2)
        points = self.meshgrid(self.bound)
        return points, self.bound

    def decode(self, rel_codes):  # return each of the patch's box, left & right, [0, 1]
        boxes = self.anchor

        if self.tanh:
            dx = torch.tanh(rel_codes[:, :, 0])
            ds = torch.relu(torch.tanh(rel_codes[:, :, 1]) + self.S_bias)

        else:
            dx = rel_codes[:, :, 0]
            ds = torch.relu(rel_codes[:, :, 1] + self.S_bias)

        pred_boxes = torch.zeros_like(rel_codes)
        ref_x = boxes.view(1, boxes.shape[0])

        # dx, ds: (bs, patch_count, 1)
        # ref_x: (1, patch_count)
        pred_boxes[:, :, 0] = dx + ref_x - ds
        pred_boxes[:, :, 1] = dx + ref_x + ds
        pred_boxes /= self.seq_len - 1

        pred_boxes = pred_boxes.clamp_(min=0.0, max=1.0)
        return pred_boxes

    def meshgrid(self, boxes):  # Input: # (bs, patch_count, 2)
        B = boxes.shape[0]
        channel_boxes = torch.zeros_like(boxes)
        channel_boxes[:, :, 1] = 1.0

        xs = torch.nn.functional.interpolate(
            boxes, size=self.patch_size, mode="linear", align_corners=True
        )
        ys = torch.nn.functional.interpolate(
            channel_boxes, size=self.channels, mode="linear", align_corners=True
        )
        # xs: [bs, patch_count, patch_size]  ys: [bs, patch_count, channels(also feats)]

        xs = xs.unsqueeze(2).expand(B, self.patch_count, self.channels, self.patch_size)
        ys = ys.unsqueeze(3).expand(B, self.patch_count, self.channels, self.patch_size)

        grid = torch.stack([xs, ys], dim=-1)
        return grid  # [bs, patch_count, channel, patch_size, 2]


class OffsetPredictor(nn.Module):
    def __init__(self, in_feats, patch_size, stride, act="gelu", mod=0):
        super().__init__()
        self.mod = mod
        self.stride = stride
        self.in_feats = in_feats
        self.patch_size = patch_size

        if mod == 0:
            self.offset_predictor = nn.Sequential(
                nn.Conv1d(in_feats, 64, patch_size, stride=stride, padding=0),
                get_activation_fn(act),
                nn.Conv1d(64, 2, 1, 1, padding=0),
            )
        elif mod == 1:  # Single Conv
            self.offset_predictor = nn.Sequential(
                nn.Conv1d(in_feats, 2, patch_size, stride=stride, padding=0),
            )
        elif mod == 2:  # MLP
            # channel independence
            in_1, in_2, in_3 = (
                patch_size * in_feats,
                2 * (patch_size * in_feats) // 3,
                (patch_size * in_feats) // 3,
            )
            out_1, out_2, out_3 = in_2, in_3, 2

            print(in_1, in_2, in_3)

            self.offset_predictor = nn.Sequential(
                nn.Linear(in_1, out_1),
                get_activation_fn(act),
                nn.Linear(in_2, out_2),
                get_activation_fn(act),
                nn.Linear(in_3, out_3),
            )

    def forward(self, X):
        # X: (bs, channel, seq_len)
        if self.mod in [0, 1]:
            pred_offset = self.offset_predictor(X).permute(0, 2, 1)
        else:
            # print('shape:', patch_X.reshape(patch_X.shape[0], -1).shape)
            patch_X = nn.functional.unfold(
                X.unsqueeze(1),
                kernel_size=(self.in_feats, self.patch_size),
                stride=(1, self.stride),
            )
            pred_offset = self.offset_predictor(patch_X.permute(0, 2, 1))

        return pred_offset  # (bs, patch_count, 2)


class DeformablePatch(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        seq_len,
        patch_size,
        stride,
        padding_tp=None,
        norm="batch",
        act="gelu",
        offset_mod=0,
    ):
        super().__init__()

        if padding_tp == "same":
            stride = 1
            self.patch_count = seq_len
            l_pad = 1 * (patch_size - 1) // 2
            r_pad = 1 * (patch_size - 1) - l_pad
            n_padding = l_pad + r_pad

        else:
            n_stride = (seq_len - patch_size) // stride + 1
            n_padding = n_stride * stride + patch_size - seq_len
            self.patch_count = n_stride + 1

        self.n_padding = n_padding

        self.patch_size = patch_size
        self.in_feats, self.out_feats = in_feats, out_feats

        self.dropout = nn.Dropout(0.1)
        self.new_len = seq_len + n_padding

        # offset predictor
        self.offset_predictor = OffsetPredictor(
            in_feats, patch_size, stride, act=act, mod=offset_mod
        )

        self.box_coder = BoxCoder(
            self.patch_count, stride, patch_size, self.new_len, in_feats
        )

        # output layers
        self.output_conv = nn.Conv2d(
            1, self.out_feats, (self.in_feats, self.patch_size)
        )
        self.norm_tp = norm
        self.output_act = get_activation_fn(act)
        self.norm = (
            nn.LayerNorm(self.out_feats)
            if norm == "layer"
            else nn.BatchNorm1d(self.out_feats)
        )

    def get_sampling_location(self, X):  # Input: (bs, channel, window)
        """
        Get sampling location.

        Input shape: (bs, channel, window).
        Sampling location shape: [bs, patch_count, channel, patch_size, 2].
        """
        # get offset
        pred_offset = self.offset_predictor(X)
        sampling_locations, bound = self.box_coder(pred_offset)
        return sampling_locations, bound

    def forward(self, X, return_bound=False):  # Input: (bs, window, channel)
        X = X.permute(0, 2, 1)
        X = nn.functional.pad(X, (0, self.n_padding), mode="constant", value=0)

        # Consider the X as img.shape: (B, C, H, W) <--> (bs,1,channel,padded_window)
        img = X.unsqueeze(1)
        B = img.shape[0]

        sampling_locations, bound = self.get_sampling_location(
            X
        )  # sampling_locations: [bs, patch_count, channel, patch_size, 2]
        sampling_locations = sampling_locations.view(
            B, self.patch_count * self.in_feats, self.patch_size, 2
        )

        # print('sampling_locations: ', sampling_locations.shape)
        sampling_locations = (sampling_locations - 0.5) * 2  # location map: [-1, 1]
        output = nn.functional.grid_sample(img, sampling_locations, align_corners=True)
        output = output.view(
            B, self.patch_count, self.in_feats, self.patch_size
        )  # (B, patch_count, channel, patch_size)

        # output_proj
        output = output.permute(0, 1, 3, 2).contiguous()  #
        output = output.view(B * self.patch_count, 1, self.in_feats, self.patch_size)
        output = self.output_conv(output)  # (bs*patch_count, out_feats, 1, 1)
        output = output.view(B, self.patch_count, self.out_feats)

        output = self.output_act(output) if self.output_act is not None else output
        if self.norm_tp == "layer":
            output = self.norm(output).permute(0, 2, 1)
        else:
            output = self.norm(output.permute(0, 2, 1))

        if return_bound:
            return output, [X, bound]
        else:
            return output
