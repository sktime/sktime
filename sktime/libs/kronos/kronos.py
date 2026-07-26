"""Kronos model, tokenizer, and predictor implementation."""

import numpy as np
import pandas as pd

from sktime.libs.kronos.module import (
    BSQuantizer,
    DependencyAwareLayer,
    DualHead,
    HierarchicalEmbedding,
    RMSNorm,
    TemporalEmbedding,
    TransformerBlock,
)
from sktime.utils.dependencies import _safe_import

trange = _safe_import("tqdm.trange")
torch = _safe_import("torch")
nn = _safe_import("torch.nn", return_object="None")
F = _safe_import("torch.nn.functional")

if nn is None:

    class _DummyModule:
        pass

    class nn:
        Module = _DummyModule


PyTorchModelHubMixin = _safe_import(
    "huggingface_hub.PyTorchModelHubMixin",
    pkg_name="huggingface_hub",
    return_object="None",
)

if PyTorchModelHubMixin is None:

    class PyTorchModelHubMixin:
        def __init_subclass__(cls, *args, **kwargs):
            pass


__all__ = [
    "Kronos",
    "KronosPredictor",
    "KronosTokenizer",
    "auto_regressive_inference",
    "calc_time_stamps",
    "sample_from_logits",
    "top_k_top_p_filtering",
]


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """Tokenize K-line features with binary spherical quantization."""

    def __init__(
        self,
        d_in,
        d_model,
        n_heads,
        ff_dim,
        n_enc_layers,
        n_dec_layers,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        s1_bits,
        s2_bits,
        beta,
        gamma0,
        gamma,
        zeta,
        group_size,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = (
            s1_bits + s2_bits
        )  # Total dimension of the codebook after quantization
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder Transformer Blocks
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.enc_layers - 1)
            ]
        )
        # Decoder Transformer Blocks
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.dec_layers - 1)
            ]
        )
        self.quant_embed = nn.Linear(
            in_features=self.d_model, out_features=self.codebook_dim
        )  # Linear layer before quantization
        self.post_quant_embed_pre = nn.Linear(
            in_features=self.s1_bits, out_features=self.d_model
        )  # Linear layer after quantization (pre part - s1 bits)
        self.post_quant_embed = nn.Linear(
            in_features=self.codebook_dim, out_features=self.d_model
        )  # Linear layer after quantization (full codebook)
        self.tokenizer = BSQuantizer(
            self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size
        )  # BSQuantizer module

    def forward(self, x):
        """Run tokenizer reconstruction and quantization."""
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z)  # (B, T, codebook)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[
            :, :, : self.s1_bits
        ]  # Extract the first part of quantized representation (s1_bits)
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        # Decoder layers (for pre part - s1 bits)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # Decoder layers (for full codebook)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        """Convert indices to scaled bit representations."""
        if half:
            x1 = x[0]  # Assuming x is a tuple of indices if half is True
            x2 = x[1]
            mask = 2 ** torch.arange(
                self.codebook_dim // 2, device=x1.device, dtype=torch.long
            )  # Create a mask for bit extraction
            x1 = (x1.unsqueeze(-1) & mask) != 0  # Extract bits for the first half
            x2 = (x2.unsqueeze(-1) & mask) != 0  # Extract bits for the second half
            x = torch.cat([x1, x2], dim=-1)  # Concatenate the bit representations
        else:
            mask = 2 ** torch.arange(
                self.codebook_dim, device=x.device, dtype=torch.long
            )  # Create a mask for bit extraction
            x = (x.unsqueeze(-1) & mask) != 0  # Extract bits

        x = x.float() * 2 - 1  # Convert boolean to bipolar (-1, 1)
        q_scale = 1.0 / (self.codebook_dim**0.5)  # Scaling factor
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """Encode input data into quantized indices."""
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(
            z, half=half, collect_metrics=False
        )
        return z_indices

    def decode(self, x, half=False):
        """Decode quantized indices back to the input data space."""
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Kronos(nn.Module, PyTorchModelHubMixin):
    """Kronos autoregressive token model."""

    def __init__(
        self,
        s1_bits,
        s2_bits,
        n_layers,
        d_model,
        n_heads,
        ff_dim,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        token_dropout_p,
        learn_te,
    ):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2**self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model**-0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        s1_ids,
        s2_ids,
        stamp=None,
        padding_mask=None,
        use_teacher_forcing=False,
        s1_targets=None,
    ):
        """Run a full forward pass over s1 and s2 token IDs."""
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1
            ).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(
            x, sibling_embed, key_padding_mask=padding_mask
        )  # Dependency Aware Layer: Condition on s1 embeddings
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """Decode only the s1 tokens."""
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """Decode s2 tokens conditioned on context and s1 tokens."""
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter logits using top-k and/or nucleus top-p filtering."""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep. The first one is added below.
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift right to keep the first token above the threshold.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(
    logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True
):
    """Sample token IDs from logits."""
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = torch.topk(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x


def auto_regressive_inference(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=5,
    verbose=False,
):
    """Generate future token reconstructions autoregressively."""
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = (
            x.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x.size(1), x.size(2))
            .to(device)
        )
        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x_stamp.size(1), x_stamp.size(2))
            .to(device)
        )
        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, y_stamp.size(1), y_stamp.size(2))
            .to(device)
        )

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][
                :, start_idx : start_idx + buffer_len
            ]
            post_buffer[:, :buffer_len] = x_token[1][
                :, start_idx : start_idx + buffer_len
            ]

        if verbose:
            ran = trange
        else:
            ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(
                input_tokens[0], input_tokens[1], current_stamp
            )
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(
                s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(
                s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        return preds


def calc_time_stamps(x_timestamp):
    """Calculate Kronos calendar features from timestamps."""
    time_df = pd.DataFrame()
    time_df["minute"] = x_timestamp.dt.minute
    time_df["hour"] = x_timestamp.dt.hour
    time_df["weekday"] = x_timestamp.dt.weekday
    time_df["day"] = x_timestamp.dt.day
    time_df["month"] = x_timestamp.dt.month
    return time_df


class KronosPredictor:
    """Preprocess OHLC data and run Kronos inference."""

    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ["open", "high", "low", "close"]
        self.vol_col = "volume"
        self.amt_vol = "amount"
        self.time_cols = ["minute", "hour", "weekday", "day", "month"]

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(
        self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose
    ):
        """Generate normalized predictions from normalized inputs."""
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(
            self.device
        )
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(
            self.device
        )

        preds = auto_regressive_inference(
            self.tokenizer,
            self.model,
            x_tensor,
            x_stamp_tensor,
            y_stamp_tensor,
            self.max_context,
            pred_len,
            self.clip,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
        )
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(
        self,
        df,
        x_timestamp,
        y_timestamp,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    ):
        """Predict one OHLC time series."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0  # Fill missing volume with zeros
            df[self.amt_vol] = 0.0  # Fill missing amount with zeros
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError(
                "Input DataFrame contains NaN values in price or volume columns."
            )

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(
            x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose
        )

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(
            preds,
            columns=self.price_cols + [self.vol_col, self.amt_vol],
            index=y_timestamp,
        )
        return pred_df

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    ):
        """Predict multiple OHLC time series in one batch."""
        # Basic validation
        if (
            not isinstance(df_list, (list, tuple))
            or not isinstance(x_timestamp_list, (list, tuple))
            or not isinstance(y_timestamp_list, (list, tuple))
        ):
            raise ValueError(
                "df_list, x_timestamp_list, and y_timestamp_list must be list "
                "or tuple types."
            )
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError(
                "df_list, x_timestamp_list, and y_timestamp_list must have "
                "consistent lengths."
            )

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(
                    f"DataFrame at index {i} is missing price columns "
                    f"{self.price_cols}."
                )

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            has_nan = (
                df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any()
            )
            if has_nan:
                raise ValueError(
                    f"DataFrame at index {i} contains NaN values in price "
                    "or volume columns."
                )

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(
                np.float32
            )
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(
                    f"Inconsistent lengths at index {i}: x has {x.shape[0]} "
                    f"vs x_stamp has {x_stamp.shape[0]}."
                )
            if y_stamp.shape[0] != pred_len:
                raise ValueError(
                    f"y_timestamp length at index {i} should equal "
                    f"pred_len={pred_len}, got {y_stamp.shape[0]}."
                )

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        # Require consistent historical and prediction lengths for batch processing.
        if len(set(seq_lens)) != 1:
            raise ValueError(
                "Parallel prediction requires consistent historical lengths, "
                f"got: {seq_lens}"
            )
        if len(set(y_lens)) != 1:
            raise ValueError(
                "Parallel prediction requires consistent prediction lengths, "
                f"got: {y_lens}"
            )

        x_batch = np.stack(x_list, axis=0).astype(np.float32)  # (B, seq_len, feat)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(
            np.float32
        )  # (B, seq_len, time_feat)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(
            np.float32
        )  # (B, pred_len, time_feat)

        preds = self.generate(
            x_batch,
            x_stamp_batch,
            y_stamp_batch,
            pred_len,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
        )
        # preds: (B, pred_len, feat)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(
                preds_i,
                columns=self.price_cols + [self.vol_col, self.amt_vol],
                index=y_timestamp_list[i],
            )
            pred_dfs.append(pred_df)

        return pred_dfs
