import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
import sys

from tqdm import trange

sys.path.append("../")
from model.module import *


class WindFMTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    WindFMTokenizer module for tokenizing input data using a hybrid quantization approach.

    This tokenizer utilizes a combination of encoder and decoder Transformer blocks
    along with the Binary Spherical Quantization (BSQuantizer) to compress and decompress input data.

    Args:
           d_in (int): Input dimension.
           d_model (int): Model dimension.
           n_heads (int): Number of attention heads.
           ff_dim (int): Feed-forward dimension.
           n_enc_layers (int): Number of encoder layers.
           n_dec_layers (int): Number of decoder layers.
           ffn_dropout_p (float): Dropout probability for feed-forward networks.
           attn_dropout_p (float): Dropout probability for attention mechanisms.
           resid_dropout_p (float): Dropout probability for residual connections.
           s1_bits (int): Number of bits for the pre token in BSQuantizer.
           s2_bits (int): Number of bits for the post token in BSQuantizer.
           beta (float): Beta parameter for BSQuantizer.
           gamma0 (float): Gamma0 parameter for BSQuantizer.
           gamma (float): Gamma parameter for BSQuantizer.
           zeta (float): Zeta parameter for BSQuantizer.
           group_size (int): Group size parameter for BSQuantizer.

    """

    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers, ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):

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
        self.codebook_dim = s1_bits + s2_bits # Total dimension of the codebook after quantization
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder Transformer Blocks
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        # Decoder Transformer Blocks
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim) # Linear layer before quantization
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model) # Linear layer after quantization (pre part - s1 bits)
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model) # Linear layer after quantization (full codebook)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size) # BSQuantizer module

    def forward(self, x):
        """
        Forward pass of the WindFMTokenizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            tuple: A tuple containing:
                - tuple: (z_pre, z) - Reconstructed outputs from decoder with s1_bits and full codebook respectively,
                         both of shape (batch_size, seq_len, d_in).
                - torch.Tensor: bsq_loss - Loss from the BSQuantizer.
                - torch.Tensor: quantized - Quantized representation from BSQuantizer.
                - torch.Tensor: z_indices - Indices from the BSQuantizer.
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z) # (B, T, codebook)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits] # Extract the first part of quantized representation (s1_bits)
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
        """
        Converts indices to bit representations and scales them.

        Args:
            x (torch.Tensor): Indices tensor.
            half (bool, optional): Whether to process only half of the codebook dimension. Defaults to False.

        Returns:
            torch.Tensor: Bit representation tensor.
        """
        if half:
            x1 = x[0] # Assuming x is a tuple of indices if half is True
            x2 = x[1]
            mask = 2 ** torch.arange(self.codebook_dim//2, device=x1.device, dtype=torch.long) # Create a mask for bit extraction
            x1 = (x1.unsqueeze(-1) & mask) != 0 # Extract bits for the first half
            x2 = (x2.unsqueeze(-1) & mask) != 0 # Extract bits for the second half
            x = torch.cat([x1, x2], dim=-1) # Concatenate the bit representations
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long) # Create a mask for bit extraction
            x = (x.unsqueeze(-1) & mask) != 0 # Extract bits

        x = x.float() * 2 - 1 # Convert boolean to bipolar (-1, 1)
        q_scale = 1. / (self.codebook_dim ** 0.5) # Scaling factor
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """
        Encodes the input data into quantized indices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).
            half (bool, optional): Whether to use half quantization in BSQuantizer. Defaults to False.

        Returns:
            torch.Tensor: Quantized indices from BSQuantizer.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z, half)
        return z_indices

    def decode(self, x, half=False):
        """
        Decodes quantized indices back to the input data space.

        Args:
            x (torch.Tensor): Quantized indices tensor.
            half (bool, optional): Whether the indices were generated with half quantization. Defaults to False.

        Returns:
            torch.Tensor: Reconstructed output tensor of shape (batch_size, seq_len, d_in).
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class WindFM(nn.Module, PyTorchModelHubMixin):
    """
    WindFM Model.

    Args:
        s1_bits (int): Number of bits for pre tokens.
        s2_bits (int): Number of bits for post tokens.
        n_layers (int): Number of Transformer blocks.
        d_model (int): Dimension of the model's embeddings and hidden states.
        n_heads (int): Number of attention heads in the MultiheadAttention layers.
        ff_dim (int): Dimension of the feedforward network in the Transformer blocks.
        ffn_dropout_p (float): Dropout probability for the feedforward network.
        attn_dropout_p (float): Dropout probability for the attention layers.
        resid_dropout_p (float): Dropout probability for residual connections.
        token_dropout_p (float): Dropout probability for token embeddings.
        learn_te (bool): Whether to use learnable temporal embeddings.
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te):
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

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = FourierTemporalEmbedding(self.d_model)
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])
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
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing for s1 decoding. Defaults to False.
            s1_targets (torch.Tensor, optional): Target s1 token IDs for teacher forcing. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - s2_logits: Logits for s2 token predictions, conditioned on s1. Shape: [batch_size, seq_len, s2_vocab_size]
        """
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
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask) # Dependency Aware Layer: Condition on s1 embeddings
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Decodes only the s1 tokens.

        This method performs a forward pass to predict only s1 tokens. It returns the s1 logits
        and the context representation from the Transformer, which can be used for subsequent s2 decoding.

        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - context: Context representation from the Transformer. Shape: [batch_size, seq_len, d_model]
        """
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
        """
        Decodes the s2 tokens, conditioned on the context and s1 tokens.

        This method decodes s2 tokens based on a pre-computed context representation (typically from `decode_s1`)
        and the s1 token IDs. It uses the dependency-aware layer and the conditional s2 head to predict s2 tokens.

        Args:
            context (torch.Tensor): Context representation from the transformer (output of decode_s1).
                                     Shape: [batch_size, seq_len, d_model]
            s1_ids (torch.torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            torch.Tensor: s2 logits. Shape: [batch_size, seq_len, s2_vocab_size]
        """
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
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x


def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_lookback, pred_len, clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5, verbose=False):
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        full_seq_len = initial_seq_len + pred_len
        device = x.device
        x = torch.clip(x, -clip, clip)

        x_repeated = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
        x_stamp_repeated = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2))
        y_stamp_repeated = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2))

        full_stamps = torch.cat([x_stamp_repeated, y_stamp_repeated], dim=1).to(device)

        initial_x_token = tokenizer.encode(x_repeated, half=True)

        full_x_token_s1 = torch.zeros(
            (batch_size * sample_count, full_seq_len),
            dtype=initial_x_token[0].dtype,
            device=device
        )
        full_x_token_s2 = torch.zeros(
            (batch_size * sample_count, full_seq_len),
            dtype=initial_x_token[1].dtype,
            device=device
        )

        full_x_token_s1[:, :initial_seq_len] = initial_x_token[0]
        full_x_token_s2[:, :initial_seq_len] = initial_x_token[1]

        if verbose:
            ran = trange
        else:
            ran = range
        for i in ran(pred_len):
            current_pos = initial_seq_len + i

            start_idx = max(0, current_pos - max_lookback)
            end_idx = current_pos

            input_tokens_s1 = full_x_token_s1[:, start_idx:end_idx]
            input_tokens_s2 = full_x_token_s2[:, start_idx:end_idx]
            current_stamp = full_stamps[:, start_idx:end_idx, :]

            s1_logits, context = model.decode_s1(input_tokens_s1, input_tokens_s2, current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            full_x_token_s1[:, current_pos] = sample_pre.squeeze(1)
            full_x_token_s2[:, current_pos] = sample_post.squeeze(1)

        final_tokens_to_decode = [
            full_x_token_s1[:, -max_lookback:].contiguous(),
            full_x_token_s2[:, -max_lookback:].contiguous()
        ]

        z = tokenizer.decode(final_tokens_to_decode, half=True)

        z = z.reshape(batch_size, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()

        return preds


def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute
    time_df['hour'] = x_timestamp.dt.hour
    time_df['weekday'] = x_timestamp.dt.weekday
    time_df['day'] = x_timestamp.dt.day
    time_df['month'] = x_timestamp.dt.month
    return time_df


class WindFMPredictor:

    def __init__(self, model, tokenizer, device="cuda:0", max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.feature_cols = ['wind_speed', 'wind_direction', 'power', 'density', 'temperature', 'pressure']
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        self.device = device

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):

        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len,
                                          self.clip, T, top_k, top_p, sample_count, verbose)
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.feature_cols):
            raise ValueError(f"Feature columns {self.feature_cols} not found in DataFrame.")

        df = df.copy()

        if df[self.feature_cols].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.feature_cols].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-6)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-6) + x_mean

        # take the power prediction
        preds = preds[:, -pred_len:, 2]
        preds = np.swapaxes(preds, 0, 1)  # (pred_len, sample_count)

        pred_df = pd.DataFrame(preds, columns=[f'pred-{i}' for i in range(preds.shape[1])], index=y_timestamp)
        return pred_df
