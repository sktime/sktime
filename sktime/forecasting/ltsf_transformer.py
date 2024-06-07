import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---> from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEmbedding, self).__init__()
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model).float()
		pe.require_grad = False

		position = torch.arange(0, max_len).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
	def __init__(self, c_in, d_model):
		super(TokenEmbedding, self).__init__()
		padding = 1 if torch.__version__ >= '1.5.0' else 2
		self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
								kernel_size=3, padding=padding, padding_mode='circular', bias=False)
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

	def forward(self, x):
		x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
		return x


class FixedEmbedding(nn.Module):
	def __init__(self, c_in, d_model):
		super(FixedEmbedding, self).__init__()

		w = torch.zeros(c_in, d_model).float()
		w.require_grad = False

		position = torch.arange(0, c_in).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

		w[:, 0::2] = torch.sin(position * div_term)
		w[:, 1::2] = torch.cos(position * div_term)

		self.emb = nn.Embedding(c_in, d_model)
		self.emb.weight = nn.Parameter(w, requires_grad=False)

	def forward(self, x):
		return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
	def __init__(self, d_model, embed_type='fixed', freq='h'):
		super(TemporalEmbedding, self).__init__()

		minute_size = 4
		hour_size = 24
		weekday_size = 7
		day_size = 32
		month_size = 13

		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		if freq == 't':
			self.minute_embed = Embed(minute_size, d_model)
		self.hour_embed = Embed(hour_size, d_model)
		self.weekday_embed = Embed(weekday_size, d_model)
		self.day_embed = Embed(day_size, d_model)
		self.month_embed = Embed(month_size, d_model)

	def forward(self, x):
		x = x.long()

		minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
		hour_x = self.hour_embed(x[:, :, 3])
		weekday_x = self.weekday_embed(x[:, :, 2])
		day_x = self.day_embed(x[:, :, 1])
		month_x = self.month_embed(x[:, :, 0])

		return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
	def __init__(self, d_model, embed_type='timeF', freq='h'):
		super(TimeFeatureEmbedding, self).__init__()

		freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
		d_inp = freq_map[freq]
		self.embed = nn.Linear(d_inp, d_model, bias=False)

	def forward(self, x):
		return self.embed(x)


class DataEmbedding(nn.Module):
	def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
		super(DataEmbedding, self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
													freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
			d_model=d_model, embed_type=embed_type, freq=freq)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark):
		x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
		return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
	def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
		super(DataEmbedding_wo_pos, self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
													freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
			d_model=d_model, embed_type=embed_type, freq=freq)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark):
		x = self.value_embedding(x) + self.temporal_embedding(x_mark)
		return self.dropout(x)

class DataEmbedding_wo_pos_temp(nn.Module):
	def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
		super(DataEmbedding_wo_pos_temp, self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
													freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
			d_model=d_model, embed_type=embed_type, freq=freq)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark):
		x = self.value_embedding(x)
		return self.dropout(x)

class DataEmbedding_wo_temp(nn.Module):
	def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
		super(DataEmbedding_wo_temp, self).__init__()

		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
													freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
			d_model=d_model, embed_type=embed_type, freq=freq)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark):
		x = self.value_embedding(x) + self.position_embedding(x)
		return self.dropout(x)



# ---> from utils.masking import TriangularCausalMask, ProbMask

import torch


class TriangularCausalMask():
	def __init__(self, B, L, device="cpu"):
		mask_shape = [B, 1, L, L]
		with torch.no_grad():
			self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

	@property
	def mask(self):
		return self._mask


class ProbMask():
	def __init__(self, B, H, L, index, scores, device="cpu"):
		_mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
		_mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
		indicator = _mask_ex[torch.arange(B)[:, None, None],
					torch.arange(H)[None, :, None],
					index, :].to(device)
		self._mask = indicator.view(scores.shape).to(device)

	@property
	def mask(self):
		return self._mask


# ---> from layers.SelfAttention_Family import FullAttention, AttentionLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
import os


class FullAttention(nn.Module):
	def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
		super(FullAttention, self).__init__()
		self.scale = scale
		self.mask_flag = mask_flag
		self.output_attention = output_attention
		self.dropout = nn.Dropout(attention_dropout)

	def forward(self, queries, keys, values, attn_mask):
		B, L, H, E = queries.shape
		_, S, _, D = values.shape
		scale = self.scale or 1. / sqrt(E)

		scores = torch.einsum("blhe,bshe->bhls", queries, keys)

		if self.mask_flag:
			if attn_mask is None:
				attn_mask = TriangularCausalMask(B, L, device=queries.device)

			scores.masked_fill_(attn_mask.mask, -np.inf)

		A = self.dropout(torch.softmax(scale * scores, dim=-1))
		V = torch.einsum("bhls,bshd->blhd", A, values)

		if self.output_attention:
			return (V.contiguous(), A)
		else:
			return (V.contiguous(), None)


class ProbAttention(nn.Module):
	def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
		super(ProbAttention, self).__init__()
		self.factor = factor
		self.scale = scale
		self.mask_flag = mask_flag
		self.output_attention = output_attention
		self.dropout = nn.Dropout(attention_dropout)

	def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
		# Q [B, H, L, D]
		B, H, L_K, E = K.shape
		_, _, L_Q, _ = Q.shape

		# calculate the sampled Q_K
		K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
		index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
		K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
		Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

		# find the Top_k query with sparisty measurement
		M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
		M_top = M.topk(n_top, sorted=False)[1]

		# use the reduced Q to calculate Q_K
		Q_reduce = Q[torch.arange(B)[:, None, None],
				torch.arange(H)[None, :, None],
				M_top, :]  # factor*ln(L_q)
		Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

		return Q_K, M_top

	def _get_initial_context(self, V, L_Q):
		B, H, L_V, D = V.shape
		if not self.mask_flag:
			# V_sum = V.sum(dim=-2)
			V_sum = V.mean(dim=-2)
			contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
		else:  # use mask
			assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
			contex = V.cumsum(dim=-2)
		return contex

	def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
		B, H, L_V, D = V.shape

		if self.mask_flag:
			attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
			scores.masked_fill_(attn_mask.mask, -np.inf)

		attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

		context_in[torch.arange(B)[:, None, None],
		torch.arange(H)[None, :, None],
		index, :] = torch.matmul(attn, V).type_as(context_in)
		if self.output_attention:
			attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
			attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
			return (context_in, attns)
		else:
			return (context_in, None)

	def forward(self, queries, keys, values, attn_mask):
		B, L_Q, H, D = queries.shape
		_, L_K, _, _ = keys.shape

		queries = queries.transpose(2, 1)
		keys = keys.transpose(2, 1)
		values = values.transpose(2, 1)

		U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
		u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

		U_part = U_part if U_part < L_K else L_K
		u = u if u < L_Q else L_Q

		scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

		# add scale factor
		scale = self.scale or 1. / sqrt(D)
		if scale is not None:
			scores_top = scores_top * scale
		# get the context
		context = self._get_initial_context(values, L_Q)
		# update the context with selected top_k queries
		context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

		return context.contiguous(), attn


class AttentionLayer(nn.Module):
	def __init__(self, attention, d_model, n_heads, d_keys=None,
				d_values=None):
		super(AttentionLayer, self).__init__()

		d_keys = d_keys or (d_model // n_heads)
		d_values = d_values or (d_model // n_heads)

		self.inner_attention = attention
		self.query_projection = nn.Linear(d_model, d_keys * n_heads)
		self.key_projection = nn.Linear(d_model, d_keys * n_heads)
		self.value_projection = nn.Linear(d_model, d_values * n_heads)
		self.out_projection = nn.Linear(d_values * n_heads, d_model)
		self.n_heads = n_heads

	def forward(self, queries, keys, values, attn_mask):
		B, L, _ = queries.shape
		_, S, _ = keys.shape
		H = self.n_heads

		queries = self.query_projection(queries).view(B, L, H, -1)
		keys = self.key_projection(keys).view(B, S, H, -1)
		values = self.value_projection(values).view(B, S, H, -1)

		out, attn = self.inner_attention(
			queries,
			keys,
			values,
			attn_mask
		)
		out = out.view(B, L, -1)

		return self.out_projection(out), attn



# ---> from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer

class ConvLayer(nn.Module):
	def __init__(self, c_in):
		super(ConvLayer, self).__init__()
		self.downConv = nn.Conv1d(in_channels=c_in,
								out_channels=c_in,
								kernel_size=3,
								padding=2,
								padding_mode='circular')
		self.norm = nn.BatchNorm1d(c_in)
		self.activation = nn.ELU()
		self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		x = self.downConv(x.permute(0, 2, 1))
		x = self.norm(x)
		x = self.activation(x)
		x = self.maxPool(x)
		x = x.transpose(1, 2)
		return x


class EncoderLayer(nn.Module):
	def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
		super(EncoderLayer, self).__init__()
		d_ff = d_ff or 4 * d_model
		self.attention = attention
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == "relu" else F.gelu

	def forward(self, x, attn_mask=None):
		new_x, attn = self.attention(
			x, x, x,
			attn_mask=attn_mask
		)
		x = x + self.dropout(new_x)

		y = x = self.norm1(x)
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))

		return self.norm2(x + y), attn


class Encoder(nn.Module):
	def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
		super(Encoder, self).__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
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


class DecoderLayer(nn.Module):
	def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
				dropout=0.1, activation="relu"):
		super(DecoderLayer, self).__init__()
		d_ff = d_ff or 4 * d_model
		self.self_attention = self_attention
		self.cross_attention = cross_attention
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == "relu" else F.gelu

	def forward(self, x, cross, x_mask=None, cross_mask=None):
		x = x + self.dropout(self.self_attention(
			x, x, x,
			attn_mask=x_mask
		)[0])
		x = self.norm1(x)

		x = x + self.dropout(self.cross_attention(
			x, cross, cross,
			attn_mask=cross_mask
		)[0])

		y = x = self.norm2(x)
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))

		return self.norm3(x + y)


class Decoder(nn.Module):
	def __init__(self, layers, norm_layer=None, projection=None):
		super(Decoder, self).__init__()
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



# ---> from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn




# ---> Autoformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Autoformer, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# ---> Transformer.py

class Transformer(nn.Module):
	"""
	Vanilla Transformer with O(L^2) complexity
	"""
	def __init__(self, configs):
		super(Transformer, self).__init__()
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention

		# Embedding
		if configs.embed_type == 0:
			self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
											configs.dropout)
			self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
										configs.dropout)
		elif configs.embed_type == 1:
			self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
													configs.dropout)
			self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
													configs.dropout)
		elif configs.embed_type == 2:
			self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
													configs.dropout)
			self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
													configs.dropout)

		elif configs.embed_type == 3:
			self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
													configs.dropout)
			self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
													configs.dropout)
		elif configs.embed_type == 4:
			self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
													configs.dropout)
			self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
													configs.dropout)
		# Encoder
		self.encoder = Encoder(
			[
				EncoderLayer(
					AttentionLayer(
						FullAttention(False, configs.factor, attention_dropout=configs.dropout,
									output_attention=configs.output_attention), configs.d_model, configs.n_heads),
					configs.d_model,
					configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation
				) for l in range(configs.e_layers)
			],
			norm_layer=torch.nn.LayerNorm(configs.d_model)
		)
		# Decoder
		self.decoder = Decoder(
			[
				DecoderLayer(
					AttentionLayer(
						FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
						configs.d_model, configs.n_heads),
					AttentionLayer(
						FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
						configs.d_model, configs.n_heads),
					configs.d_model,
					configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation,
				)
				for l in range(configs.d_layers)
			],
			norm_layer=torch.nn.LayerNorm(configs.d_model),
			projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
		)

	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
				enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

		dec_out = self.dec_embedding(x_dec, x_mark_dec)
		dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

		if self.output_attention:
			return dec_out[:, -self.pred_len:, :], attns
		else:
			return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# import torch

# # Define configuration object with necessary parameters
# class Configs:
#     def __init__(self):
#         self.pred_len = 24
#         self.output_attention = False
#         self.embed_type = 0  # Example: Use DataEmbedding
#         self.embed = "fixed"  # Example: Use DataEmbedding
#         self.enc_in = 7  # Number of encoder input features
#         self.dec_in = 7  # Number of decoder input features
#         self.d_model = 512  # Dimension of model
#         self.n_heads = 8  # Number of attention heads
#         self.d_ff = 2048  # Dimension of feedforward network
#         self.e_layers = 3  # Number of encoder layers
#         self.d_layers = 2  # Number of decoder layers
#         self.factor = 5  # Factor for ProbAttention
#         self.dropout = 0.1  # Dropout rate
#         self.activation = "relu"  # Activation function
#         self.c_out = 7  # Number of output features
#         self.freq = 'h'  # Frequency for temporal embeddings

# configs = Configs()

# # Initialize the model
# model = Transformer(configs)

# # Example input data
# x_enc = torch.rand((32, 96, 7))  # (batch_size, seq_len, num_features)
# x_mark_enc = torch.rand((32, 96, 4))  # (batch_size, seq_len, num_time_features)
# x_dec = torch.rand((32, 24, 7))  # (batch_size, pred_len, num_features)
# x_mark_dec = torch.rand((32, 24, 4))  # (batch_size, pred_len, num_time_features)

# # Forward pass
# output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
# print(output.shape)  # Expected output shape: (32, 24, 7)