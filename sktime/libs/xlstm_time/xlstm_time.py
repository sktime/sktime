# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Vendorized xLSTMTime core modules.

Based on https://github.com/muslehal/xLSTMTime.
"""

__all__ = ["mLSTMBlock", "sLSTMBlock", "xLSTM", "xLSTMBlock"]
__author__ = ["muslehal", "vedantag17"]

from sktime.utils.dependencies import _safe_import

nn = _safe_import("torch.nn")
torch = _safe_import("torch")


class sLSTMBlock(nn.Module):
    """Stabilized LSTM Block."""

    def __init__(self, input_size, hidden_size, num_heads=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Input, forget, and output gates
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_z = nn.Linear(input_size, hidden_size)

        # Recurrent connections
        self.R_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Layer normalization
        self.ln_i = nn.LayerNorm(hidden_size)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.ln_o = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)

    def forward(self, x, state=None):
        """Forward pass through the sLSTM block."""
        batch_size, seq_len, _ = x.shape

        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
            n = torch.zeros(batch_size, self.hidden_size, device=x.device)
            m = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c, n, m = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Compute gates
            i_t = torch.sigmoid(self.ln_i(self.W_i(x_t) + self.R_i(h)))
            f_t = torch.sigmoid(self.ln_f(self.W_f(x_t) + self.R_f(h)))
            o_t = torch.sigmoid(self.ln_o(self.W_o(x_t) + self.R_o(h)))
            z_t = torch.tanh(self.ln_z(self.W_z(x_t) + self.R_z(h)))

            # Update cell state with stabilization
            m = torch.max(f_t + i_t, m)
            i_t_hat = torch.exp(i_t - m)
            f_t_hat = torch.exp(f_t - m)

            c = f_t_hat * c + i_t_hat * z_t
            n = f_t_hat * n + i_t_hat

            # Stabilized cell state
            c_tilde = c / n
            h = o_t * torch.tanh(c_tilde)

            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, (h, c, n, m)


class mLSTMBlock(nn.Module):
    """Matrix LSTM Block."""

    def __init__(self, input_size, hidden_size, num_heads=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

        # Input and forget gates
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)

        # Layer normalization
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_k = nn.LayerNorm(hidden_size)
        self.ln_v = nn.LayerNorm(hidden_size)

        # Output projection
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, state=None):
        """Forward pass through the mLSTM block."""
        batch_size, seq_len, _ = x.shape

        if state is None:
            C = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
            )
            n = torch.zeros(batch_size, self.num_heads, self.head_dim, device=x.device)
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            C, n, h = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Compute projections
            q_t = self.ln_q(self.W_q(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )
            k_t = self.ln_k(self.W_k(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )
            v_t = self.ln_v(self.W_v(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )

            # Compute gates
            i_t = torch.sigmoid(self.W_i(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )
            f_t = torch.sigmoid(self.W_f(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )
            o_t = torch.sigmoid(self.W_o(x_t))

            # Update matrix memory
            C = f_t.unsqueeze(-1) * C + i_t.unsqueeze(-1) * torch.bmm(
                v_t.unsqueeze(-1), k_t.unsqueeze(-2)
            )
            n = f_t * n + i_t * k_t

            # Compute output
            h_heads = torch.bmm(C, q_t.unsqueeze(-1)).squeeze(-1) / (
                n.unsqueeze(-1) + 1e-8
            )
            h_concat = h_heads.view(batch_size, -1)
            h = o_t * self.proj(h_concat)

            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, (C, n, h)


class xLSTMBlock(nn.Module):
    """Extended LSTM Block (can be either sLSTM or mLSTM)."""

    def __init__(
        self, input_size, hidden_size, block_type="slstm", num_heads=1, dropout=0.0
    ):
        super().__init__()
        self.block_type = block_type

        if block_type == "slstm":
            self.block = sLSTMBlock(input_size, hidden_size, num_heads)
        elif block_type == "mlstm":
            self.block = mLSTMBlock(input_size, hidden_size, num_heads)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, state=None):
        """Forward pass through the xLSTM block."""
        # Residual connection
        residual = x if x.shape[-1] == self.block.hidden_size else None

        output, new_state = self.block(x, state)
        output = self.dropout(output)

        # Add residual connection if dimensions match
        if residual is not None:
            output = output + residual

        output = self.norm(output)
        return output, new_state


class xLSTM(nn.Module):
    """Extended Long Short-Term Memory Network."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        block_types=None,
        num_heads=1,
        dropout=0.0,
        output_size=1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        if block_types is None:
            block_types = ["slstm"] * num_layers

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # xLSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = hidden_size
            self.layers.append(
                xLSTMBlock(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    block_type=block_types[i % len(block_types)],
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, states=None):
        """Forward pass through the xLSTM network."""
        # Input projection
        x = self.input_proj(x)

        if states is None:
            states = [None] * self.num_layers

        new_states = []

        # Pass through xLSTM layers
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, states[i])
            new_states.append(new_state)

        # Output projection (use last timestep)
        output = self.output_proj(x[:, -1, :])

        return output
