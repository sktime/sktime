"""CNTC Neural Network for Classification and Regression in PyTorch."""

__authors__ = ["fnhirwa", "srupat"]
__all__ = ["CNTCNetworkTorch"]

from collections.abc import Callable

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


def _as_tuple(value, length, name):
    """Broadcast a scalar to a tuple, or validate an existing sequence.

    Parameters
    ----------
    value : float or sequence of float
        Value to broadcast or validate.
    length : int
        Required length of the resulting tuple.
    name : str
        Parameter name, used in error messages.

    Returns
    -------
    tuple
        Tuple of ``length`` values.
    """
    if length == 0:
        # the corresponding stack of layers is empty, so any rate is vacuous
        return ()
    if isinstance(value, (int, float)):
        return (value,) * length
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(
                f"`{name}` must be a float or a sequence of length {length}. "
                f"Found length {len(value)}."
            )
        return tuple(value)
    raise TypeError(
        f"`{name}` must be a float or a sequence of floats. "
        f"But found the type to be: {type(value)}"
    )


class _ContextualConv1d(NNModule):
    """Contextual convolutional layer, the core module of the CCNN arm.

    Implements the recurrent convolutional layer of equation (1) in [1]_. A
    feed-forward convolution of the input is computed once and held fixed,
    while a second, weight-shared convolution is applied repeatedly to the
    layer's own evolving output ``M``:

    .. code-block:: none

        M[0] = conv_feed(s)
        M[k] = BN_k(beta(conv_feed(s) + conv_rec(M[k - 1])))

    for ``k = 1 ... context_steps``. Because the recurrence reuses the same
    ``conv_rec`` weights at every step, ``context_steps`` iterations of a
    kernel of size ``n`` give an effective receptive field of
    ``(n - 1) * context_steps + 1`` while the parameter count stays constant.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of convolution filters.
    kernel_size : int
        Length of the 1D convolution window.
    context_steps : int
        Number of recurrent iterations ``K``.
    activation_hidden : callable or None
        Non-linearity ``beta`` applied at every iteration.

    Note
    ----
    Equation (1) of [1]_ writes the same symbol for the weights of
    both the input and the recurrent term. Those two convolutions map
    different channel counts (``in_channels`` and ``out_channels``
    respectively), so they cannot literally share weights; separate
    convolutions are used here, which is the standard recurrent convolutional
    layer formulation also cited in the paper.

    References
    ----------
    .. [1] Amadu Fullah Kamara, Enhong Chen, Qi Liu, Zhen Pan, Combin-
       ing Contextual Neural Networks for Time Series Classification,
       Neurocomputing (2019), doi:
       https://doi.org/10.1016/j.neucom.2019.10.113
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        context_steps,
        activation_hidden,
    ):
        super().__init__()

        if context_steps < 1:
            raise ValueError(
                f"`context_steps` must be a positive integer. Found {context_steps}."
            )

        Conv1d = _safe_import("torch.nn.Conv1d")
        BatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        ModuleList = _safe_import("torch.nn.ModuleList")

        self.context_steps = context_steps
        self.activation_hidden = activation_hidden

        # padding="same" keeps the time axis intact so that the CCNN and CLSTM
        # arms can be concatenated per time step in the concatenation stage
        self.conv_feed = Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        self.conv_rec = Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        # batch normalisation prevents the recurrent states from exploding
        self.norms = ModuleList(
            [BatchNorm1d(out_channels) for _ in range(context_steps)]
        )

    def forward(self, X):
        """Iterate the contextual convolution over ``context_steps`` steps.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, in_channels, series_length)
            Input tensor, channels first.

        Returns
        -------
        torch.Tensor of shape (batch_size, out_channels, series_length)
        """
        feed = self.conv_feed(X)
        state = feed
        for step in range(self.context_steps):
            # the first iteration is equation (2) with M initialised to the
            # input, so it costs no recurrent convolution
            if step > 0:
                state = feed + self.conv_rec(state)
            if self.activation_hidden is not None:
                state = self.activation_hidden(state)
            # BatchNorm1d needs more than one sample per channel while training
            if state.shape[0] > 1 or not self.training:
                state = self.norms[step](state)
        return state


class _CLSTMCell(NNModule):
    """Contextual LSTM cell, the core module of the CLSTM arm.

    Implements equation (4) of [1]_: a standard LSTM cell in which every gate
    additionally receives a projection of the contextual feature vector ``P``,
    derived from sliding-window means of the input series.

    .. code-block:: none

        ctx  = W_ip @ P_k + b_ip
        i, f, o, g = split(W_ih @ s_k + W_hh @ h_[k-1] + b)
        i, f, o = sigmoid(i + ctx), sigmoid(f + ctx), sigmoid(o + ctx)
        g       = tanh(g + ctx)
        c_k = f * c_[k-1] + i * g
        h_k = o * tanh(c_k)

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of LSTM cells.
    context_size : int
        Dimension of the contextual feature vector ``P``.
    """

    def __init__(self, input_size, hidden_size, context_size):
        super().__init__()

        Linear = _safe_import("torch.nn.Linear")

        self.hidden_size = hidden_size
        self.weight_ih = Linear(input_size, 4 * hidden_size)
        self.weight_hh = Linear(hidden_size, 4 * hidden_size, bias=False)
        self.weight_ip = Linear(context_size, hidden_size)

    def forward(self, X, P):
        """Run the contextual LSTM recurrence over a whole sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, series_length, input_size)
            Input sequence.
        P : torch.Tensor of shape (batch_size, series_length, context_size)
            Contextual features, one vector per time step.

        Returns
        -------
        torch.Tensor of shape (batch_size, series_length, hidden_size)
            Hidden state at every time step.
        """
        torch_sigmoid = _safe_import("torch.sigmoid")
        torch_tanh = _safe_import("torch.tanh")
        torch_stack = _safe_import("torch.stack")
        torch_zeros = _safe_import("torch.zeros")

        batch_size, series_length, _ = X.shape
        h = torch_zeros(batch_size, self.hidden_size, device=X.device, dtype=X.dtype)
        c = h.clone()

        # the context projection enters every gate, so it is computed once
        # for the whole sequence and sliced per time step
        ctx = self.weight_ip(P)

        outputs = []
        for step in range(series_length):
            gates = self.weight_ih(X[:, step]) + self.weight_hh(h)
            gate_i, gate_f, gate_o, gate_g = gates.chunk(4, dim=-1)
            ctx_step = ctx[:, step]
            gate_i = torch_sigmoid(gate_i + ctx_step)
            gate_f = torch_sigmoid(gate_f + ctx_step)
            gate_o = torch_sigmoid(gate_o + ctx_step)
            gate_g = torch_tanh(gate_g + ctx_step)
            c = gate_f * c + gate_i * gate_g
            h = gate_o * torch_tanh(c)
            outputs.append(h)

        return torch_stack(outputs, dim=1)


class CNTCNetworkTorch(NNModule):
    """Establish the network structure for CNTC in PyTorch.

    CNTC combines a Contextual Convolutional Neural Network (CCNN) and a
    Contextual Long Short-Term Memory network (CLSTM) as parallel feature
    extractors, concatenates their outputs per time step, refines them with
    self-attention and classifies with a multilayer perceptron, following [1]_.

    The four stages are:

    - **Feature extraction.** The CCNN arm stacks contextual convolutional
      layers (recurrent convolutions, equation 1 of [1]_) followed by standard
      convolutional layers. The CLSTM arm stacks contextual LSTM layers whose
      gates receive sliding-window means of the input as contextual features
      (equation 4 of [1]_). Both arms see the same input.
    - **Concatenation.** The two arms are concatenated along the feature axis,
      per time step, giving ``c_k = concat(mu_k, h_k)`` (equation 5 of [1]_).
    - **Attention.** Pooling downsamples the merged sequence, then sequential
      self-attention reweights it (equations 6 to 8 of [1]_).
    - **Multilayer perceptron.** Fully connected layers with dropout, followed
      by the output layer.

    Parameters
    ----------
    input_size : tuple of int
        Shape of the input data. Must be either of length 3 and in format
        ``(n_instances, n_dims, series_length)``, or of length 2 and in format
        ``(n_dims, series_length)``.
    num_classes : int
        Number of outputs.
    activation : callable or None, default = None
        Activation function applied to the output layer. If callable, it must
        accept and return a torch tensor. ``None`` leaves the network returning
        raw outputs, which is what ``CrossEntropyLoss`` expects.
    activation_hidden : callable or None, default = None
        Activation function applied to the hidden layers, that is the
        convolutional layers of the CCNN arm and the dense layers of the
        multilayer perceptron.
    context_filter_sizes : tuple of int, default = (16,)
        Number of filters in each contextual convolutional layer of the CCNN
        arm. The length of the tuple sets the number of such layers; any length
        >= 1 is allowed. [1]_ uses a single layer with 8, 16, 32 or 64 filters.
    context_kernel_sizes : tuple of int, default = (3,)
        Length of the 1D convolution window of each contextual convolutional
        layer. Must have the same length as ``context_filter_sizes``. Combined
        with ``context_steps``, a kernel of size ``n`` gives an effective
        receptive field of ``(n - 1) * context_steps + 1``.
    context_steps : int, default = 3
        Number of recurrent iterations ``K`` performed inside every contextual
        convolutional layer. ``context_steps=1`` reduces the layer to an
        ordinary convolution.
    context_dropout : float or tuple of float, default = 0.8
        Dropout rate applied after each contextual convolutional layer. A float
        applies the same rate to all of them, a tuple sets them individually and
        must have the same length as ``context_filter_sizes``.
    conv_filter_sizes : tuple of int, default = (8,)
        Number of filters in each standard convolutional layer, applied after
        the contextual convolutional layers. The length of the tuple sets the
        number of such layers; ``()`` disables them entirely. [1]_ uses a single
        layer with 8, 16 or 32 filters.
    conv_kernel_sizes : tuple of int, default = (3,)
        Length of the 1D convolution window of each standard convolutional
        layer. Must have the same length as ``conv_filter_sizes``.
    conv_dropout : float or tuple of float, default = 0.8
        Dropout rate applied after each standard convolutional layer. A float
        applies the same rate to all of them, a tuple sets them individually and
        must have the same length as ``conv_filter_sizes``.
    lstm_units : tuple of int, default = (8,)
        Number of cells in each contextual LSTM layer of the CLSTM arm. The
        length of the tuple sets the number of stacked layers; any length >= 1
        is allowed. [1]_ uses a single layer with 8, 16, 32 or 64 cells.
    context_window : int, default = 3
        Size of the sliding window used to build the contextual features of the
        CLSTM arm. At each time step the context vector holds the
        ``context_window`` most recent window means, so it has dimension
        ``context_window * n_dims``.
    lstm_dropout : float or tuple of float, default = 0.8
        Dropout rate applied after each contextual LSTM layer. A float applies
        the same rate to all of them, a tuple sets them individually and must
        have the same length as ``lstm_units``. [1]_ uses 0.8.
    pool_size : int, default = 2
        Size and stride of the pooling window applied to the merged sequence
        before attention. Values above 1 downsample the time axis.
        Clipped to the series length.
    pool_type : str or None, default = "max"
        Pooling to apply before attention. One of ``"max"``, ``"avg"``,
        ``"both"`` (max and average pooling concatenated along the feature
        axis), or ``None`` to disable pooling.
    pool_dropout : float, default = 0.6
        Dropout rate applied after pooling.
    attention_width : int or None, default = 10
        Width of the local attention window. ``None`` lets every time step
        attend to every other time step. [1]_ uses 8 or 10.
    attention_units : int, default = 32
        Hidden dimension of the additive attention scorer. Unused when
        ``attention_type="multiplicative"``.
    attention_type : str, default = "additive"
        Attention scoring function, either ``"additive"`` or
        ``"multiplicative"``. [1]_ scores alignments with a feedforward network,
        which corresponds to ``"additive"``.
    attention_activation : str, callable, torch.nn.Module or None, default = None
        Non-linearity applied to the attention logits before normalisation.
        ``None`` normalises the raw alignment scores with a softmax, as in
        equation (7) of [1]_.
    attention_dropout : float, default = 0.5
        Dropout rate applied after attention. [1]_ uses 0.5.
    dense_layers : tuple of int, default = (64, 64)
        Number of units in each dense layer of the multilayer perceptron. The
        length of the tuple sets the number of such layers; ``()`` connects the
        attention output straight to the output layer. [1]_ uses two layers of
        64 units.
    dense_dropout : float or tuple of float, default = (0.5, 0.8)
        Dropout rate applied after each dense layer. A float applies the same
        rate to all of them, a tuple sets them individually and must have the
        same length as ``dense_layers``. [1]_ uses 0.5 and 0.8.
    init_weights : str or None, default = "xavier_uniform"
        The method used to initialize the weights of the convolutional and
        linear layers. Supported values are ``"kaiming_uniform"``,
        ``"kaiming_normal"``, ``"xavier_uniform"``, ``"xavier_normal"``, or
        ``None`` for the default PyTorch initialization. Biases are zeroed
        whenever a method is given.
    random_state : int or None, default = None
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Amadu Fullah Kamara, Enhong Chen, Qi Liu, Zhen Pan, Combin-
       ing Contextual Neural Networks for Time Series Classification,
       Neurocomputing (2019), doi:
       https://doi.org/10.1016/j.neucom.2019.10.113
    """

    _tags = {
        "authors": __authors__,
        "maintainers": ["fnhirwa", "srupat"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    _INIT_METHODS = (
        "kaiming_uniform",
        "kaiming_normal",
        "xavier_uniform",
        "xavier_normal",
    )

    def __init__(
        self,
        input_size: tuple[int, ...],
        num_classes: int,
        activation: Callable | None = None,
        activation_hidden: Callable | None = None,
        context_filter_sizes: tuple[int, ...] = (16,),
        context_kernel_sizes: tuple[int, ...] = (3,),
        context_steps: int = 3,
        context_dropout: float | tuple[float, ...] = 0.8,
        conv_filter_sizes: tuple[int, ...] = (8,),
        conv_kernel_sizes: tuple[int, ...] = (3,),
        conv_dropout: float | tuple[float, ...] = 0.8,
        lstm_units: tuple[int, ...] = (8,),
        context_window: int = 3,
        lstm_dropout: float | tuple[float, ...] = 0.8,
        pool_size: int = 2,
        pool_type: str | None = "max",
        pool_dropout: float = 0.6,
        attention_width: int | None = 10,
        attention_units: int = 32,
        attention_type: str = "additive",
        attention_activation: str | Callable | None = None,
        attention_dropout: float = 0.5,
        dense_layers: tuple[int, ...] = (64, 64),
        dense_dropout: float | tuple[float, ...] = (0.5, 0.8),
        init_weights: str | None = "xavier_uniform",
        random_state: int | None = None,
    ):
        super().__init__()

        self._import_cache = {}
        self.input_size = input_size
        self.num_classes = num_classes
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.context_filter_sizes = context_filter_sizes
        self.context_kernel_sizes = context_kernel_sizes
        self.context_steps = context_steps
        self.context_dropout = context_dropout
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_dropout = conv_dropout
        self.lstm_units = lstm_units
        self.context_window = context_window
        self.lstm_dropout = lstm_dropout
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.pool_dropout = pool_dropout
        self.attention_width = attention_width
        self.attention_units = attention_units
        self.attention_type = attention_type
        self.attention_activation = attention_activation
        self.attention_dropout = attention_dropout
        self.dense_layers = dense_layers
        self.dense_dropout = dense_dropout
        self.init_weights = init_weights
        self.random_state = random_state

        n_dims, series_length = self._validate_input_size(input_size)
        self.n_dims = n_dims
        self.series_length = series_length

        if self.random_state is not None:
            torch_manual_seed = _safe_import("torch.manual_seed")
            torch_manual_seed(self.random_state)

        if self.context_window < 1:
            raise ValueError(
                "`context_window` must be a positive integer. "
                f"Found {self.context_window}."
            )

        if init_weights is not None and init_weights not in self._INIT_METHODS:
            raise ValueError(
                "`init_weights` must be None or one of "
                f"{', '.join(self._INIT_METHODS)}. Found {init_weights}."
            )

        Dropout = _safe_import("torch.nn.Dropout")
        ModuleList = _safe_import("torch.nn.ModuleList")

        # CCNN arm: contextual convolutional layers
        self._validate_conv_spec(
            context_filter_sizes,
            context_kernel_sizes,
            "context_filter_sizes",
            "context_kernel_sizes",
            allow_empty=False,
        )
        context_drops = _as_tuple(
            context_dropout, len(context_filter_sizes), "context_dropout"
        )
        self.context_convs = ModuleList()
        self.context_drops = ModuleList()
        in_channels = n_dims
        for filters, kernel, rate in zip(
            context_filter_sizes, context_kernel_sizes, context_drops
        ):
            self.context_convs.append(
                _ContextualConv1d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel,
                    context_steps=context_steps,
                    activation_hidden=activation_hidden,
                )
            )
            self.context_drops.append(Dropout(p=rate))
            in_channels = filters

        # CCNN arm: standard convolutional layers
        self._validate_conv_spec(
            conv_filter_sizes,
            conv_kernel_sizes,
            "conv_filter_sizes",
            "conv_kernel_sizes",
            allow_empty=True,
        )
        conv_drops = _as_tuple(conv_dropout, len(conv_filter_sizes), "conv_dropout")
        Conv1d = _safe_import("torch.nn.Conv1d")
        BatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        self.convs = ModuleList()
        self.conv_norms = ModuleList()
        self.conv_drops = ModuleList()
        for filters, kernel, rate in zip(
            conv_filter_sizes, conv_kernel_sizes, conv_drops
        ):
            self.convs.append(
                Conv1d(in_channels, filters, kernel_size=kernel, padding="same")
            )
            self.conv_norms.append(BatchNorm1d(filters))
            self.conv_drops.append(Dropout(p=rate))
            in_channels = filters
        ccnn_features = in_channels

        # CLSTM arm: contextual LSTM layers
        if not isinstance(lstm_units, (list, tuple)) or len(lstm_units) < 1:
            raise ValueError(
                f"`lstm_units` must be a non-empty tuple of ints. Found {lstm_units}."
            )
        lstm_drops = _as_tuple(lstm_dropout, len(lstm_units), "lstm_dropout")
        self.context_size = context_window * n_dims
        AvgPool1d = _safe_import("torch.nn.AvgPool1d")
        self.context_mean = AvgPool1d(context_window, stride=1)
        self.lstm_cells = ModuleList()
        self.lstm_drops = ModuleList()
        lstm_in = n_dims
        for units, rate in zip(lstm_units, lstm_drops):
            self.lstm_cells.append(
                _CLSTMCell(
                    input_size=lstm_in,
                    hidden_size=units,
                    context_size=self.context_size,
                )
            )
            self.lstm_drops.append(Dropout(p=rate))
            lstm_in = units
        clstm_features = lstm_in

        # concatenation stage merges both arms along the feature axis
        merged_features = ccnn_features + clstm_features

        # attention stage, preceded by pooling
        if pool_type not in ("max", "avg", "both", None):
            raise ValueError(
                "`pool_type` must be one of 'max', 'avg', 'both' or None. "
                f"Found {pool_type}."
            )
        self.max_pool = None
        self.avg_pool = None
        if pool_type is None:
            pooled_length = series_length
            attention_features = merged_features
        else:
            if pool_size < 1:
                raise ValueError(
                    f"`pool_size` must be a positive integer. Found {pool_size}."
                )
            # a pooling window longer than the series would empty the time axis
            effective_pool = min(pool_size, series_length)
            pooled_length = series_length // effective_pool
            attention_features = merged_features * (2 if pool_type == "both" else 1)
            if pool_type in ("max", "both"):
                MaxPool1d = _safe_import("torch.nn.MaxPool1d")
                self.max_pool = MaxPool1d(effective_pool, stride=effective_pool)
            if pool_type in ("avg", "both"):
                AvgPool1d = _safe_import("torch.nn.AvgPool1d")
                self.avg_pool = AvgPool1d(effective_pool, stride=effective_pool)
        self.pool_drop = Dropout(p=pool_dropout)

        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        self.attention = SeqSelfAttentionTorch(
            input_dim=attention_features,
            units=attention_units,
            attention_width=attention_width,
            attention_type=attention_type,
            attention_activation=attention_activation,
        )
        self.attention_drop = Dropout(p=attention_dropout)

        # multilayer perceptron stage
        if not isinstance(dense_layers, (list, tuple)):
            raise TypeError(
                f"`dense_layers` must be a tuple of ints. Found {type(dense_layers)}."
            )
        dense_drops = _as_tuple(dense_dropout, len(dense_layers), "dense_dropout")
        Linear = _safe_import("torch.nn.Linear")
        self.denses = ModuleList()
        self.dense_drops = ModuleList()
        in_features = pooled_length * attention_features
        for units, rate in zip(dense_layers, dense_drops):
            self.denses.append(Linear(in_features, units))
            self.dense_drops.append(Dropout(p=rate))
            in_features = units

        self.out = Linear(in_features, num_classes)

        if self.init_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _validate_input_size(input_size):
        """Extract ``n_dims`` and ``series_length`` from ``input_size``."""
        if isinstance(input_size, (list, tuple)):
            if len(input_size) == 3:
                return int(input_size[1]), int(input_size[2])
            if len(input_size) == 2:
                return int(input_size[0]), int(input_size[1])
            raise ValueError(
                "`input_size` must be a tuple of length 3, in format "
                "(n_instances, n_dims, series_length), or of length 2, in "
                f"format (n_dims, series_length). Found length of {len(input_size)}"
            )
        raise TypeError(
            "`input_size` should be a tuple carrying the series length, either "
            "(n_instances, n_dims, series_length) or (n_dims, series_length). "
            f"But found the type to be: {type(input_size)}"
        )

    @staticmethod
    def _validate_conv_spec(filters, kernels, filters_name, kernels_name, allow_empty):
        """Validate that a filter and kernel specification pair is consistent."""
        for value, name in ((filters, filters_name), (kernels, kernels_name)):
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, int) for item in value
            ):
                raise TypeError(f"`{name}` must be a tuple of ints. Found {value}.")
        if len(filters) != len(kernels):
            raise ValueError(
                f"`{filters_name}` and `{kernels_name}` must be of the same length. "
                f"Found {len(filters)} and {len(kernels)}."
            )
        if not allow_empty and len(filters) < 1:
            raise ValueError(f"`{filters_name}` must have length >= 1.")

    def _torch_op(self, import_path):
        """Lazy import and cache torch ops used in the forward pass."""
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]

    def _init_weights(self, module):
        """Apply the configured initialization to a module.

        Parameters
        ----------
        module : torch.nn.Module
            Input module on which to apply the initialization.
        """
        Conv1d = _safe_import("torch.nn.Conv1d")
        Linear = _safe_import("torch.nn.Linear")

        kaiming_uniform_ = _safe_import("torch.nn.init.kaiming_uniform_")
        kaiming_normal_ = _safe_import("torch.nn.init.kaiming_normal_")
        xavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")
        xavier_normal_ = _safe_import("torch.nn.init.xavier_normal_")

        # linear layers are initialized alongside the convolutions, since the
        # CLSTM cells and the multilayer perceptron are built from them
        if isinstance(module, (Conv1d, Linear)):
            if self.init_weights == "kaiming_uniform":
                kaiming_uniform_(module.weight, nonlinearity="relu")

            elif self.init_weights == "kaiming_normal":
                kaiming_normal_(module.weight, nonlinearity="relu")

            elif self.init_weights == "xavier_uniform":
                xavier_uniform_(module.weight)

            elif self.init_weights == "xavier_normal":
                xavier_normal_(module.weight)

            if module.bias is not None:
                module.bias.data.zero_()

    def _build_context(self, X):
        """Build the contextual features of the CLSTM arm.

        Computes sliding-window means of the input series and, for every time
        step, gathers the ``context_window`` most recent means into a single
        context vector.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, series_length, n_dims)
            Input sequence.

        Returns
        -------
        torch.Tensor of shape (batch_size, series_length, context_window * n_dims)
        """
        pad = self._torch_op("torch.nn.functional.pad")

        window = self.context_window
        batch_size, series_length, n_dims = X.shape

        # replicate padding keeps the means causal and defined at the boundary
        channels_first = X.transpose(1, 2)
        padded = pad(channels_first, (window - 1, 0), mode="replicate")
        means = self.context_mean(padded)

        # gather the trailing `window` means at every time step
        padded_means = pad(means, (window - 1, 0), mode="replicate")
        context = padded_means.unfold(dimension=2, size=window, step=1)
        context = context.permute(0, 2, 1, 3)
        return context.reshape(batch_size, series_length, n_dims * window)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, series_length, n_dims)
            Input tensor containing the time series data.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes), or of shape
        (batch_size,) when ``num_classes=1``, that is for regression.
        """
        if isinstance(X, np.ndarray):
            torch_from_numpy = self._torch_op("torch.from_numpy")
            X = torch_from_numpy(X).float()

        torch_cat = self._torch_op("torch.cat")

        # CCNN arm, channels first for the convolutions
        x_ccnn = X.transpose(1, 2)
        for conv, drop in zip(self.context_convs, self.context_drops):
            x_ccnn = drop(conv(x_ccnn))
        for conv, norm, drop in zip(self.convs, self.conv_norms, self.conv_drops):
            x_ccnn = conv(x_ccnn)
            if self.activation_hidden is not None:
                x_ccnn = self.activation_hidden(x_ccnn)
            if x_ccnn.shape[0] > 1 or not self.training:
                x_ccnn = norm(x_ccnn)
            x_ccnn = drop(x_ccnn)
        x_ccnn = x_ccnn.transpose(1, 2)

        # CLSTM arm, sharing the contextual features across stacked layers
        context = self._build_context(X)
        x_clstm = X
        for cell, drop in zip(self.lstm_cells, self.lstm_drops):
            x_clstm = drop(cell(x_clstm, context))

        # concatenation stage, merging both arms per time step
        merged = torch_cat([x_ccnn, x_clstm], dim=-1)

        # attention stage, pooling first
        if self.pool_type is not None:
            pooled = merged.transpose(1, 2)
            if self.pool_type == "max":
                pooled = self.max_pool(pooled)
            elif self.pool_type == "avg":
                pooled = self.avg_pool(pooled)
            else:
                pooled = torch_cat(
                    [self.max_pool(pooled), self.avg_pool(pooled)], dim=1
                )
            merged = pooled.transpose(1, 2)
        merged = self.pool_drop(merged)

        attended = self.attention_drop(self.attention(merged))

        # multilayer perceptron stage
        out = attended.reshape(attended.shape[0], -1)
        for dense, drop in zip(self.denses, self.dense_drops):
            out = dense(out)
            if self.activation_hidden is not None:
                out = self.activation_hidden(out)
            out = drop(out)

        out = self.out(out)
        if self.activation is not None:
            out = self.activation(out)

        if self.num_classes == 1:  # regression, match the target's 1D shape
            out = out.squeeze(1)
        return out
