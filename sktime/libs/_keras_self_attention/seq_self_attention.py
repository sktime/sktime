"""Sequential self-attention layer."""

from sktime.utils.dependencies import _safe_import

keras = _safe_import("tensorflow.keras")
K = _safe_import("tensorflow.keras.backend")


class SeqSelfAttention(keras.layers.Layer):
    """Sequential self-attention layer.

    For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf

    Parameters
    ----------
    units: int, optional (default=32)
        The dimension of the vectors that used to calculate the attention weights.
    attention_width: int, optional (default=None)
        The width of local attention.
    attention_type: str, optional (default='additive')
        'additive' or 'multiplicative'.
    return_attention: bool, optional (default=False)
        Whether to return the attention weights for visualization.
    history_only: bool, optional (default=False)
        Only use historical pieces of data.
    kernel_initializer: str, optional (default='glorot_normal')
        The initializer for weight matrices.
    bias_initializer: str, optional (default='zeros')
        The initializer for biases.
    kernel_regularizer: str, optional (default=None)
        The regularization for weight matrices.
    bias_regularizer: str, optional (default=None)
        The regularization for biases.
    kernel_constraint: str, optional (default=None)
        The constraint for weight matrices.
    bias_constraint: str, optional (default=None)
        The constraint for biases.
    use_additive_bias: bool, optional (default=True)
        Whether to use bias while calculating the relevance of inputs features
        in additive mode.
    use_attention_bias: bool, optional (default=True)
        Whether to use bias while calculating the weights of attention.
    attention_activation: str, optional (default=None)
        The activation used for calculating the weights of attention.
    attention_regularizer_weight: float, optional (default=0.0)
        The weights of attention regularizer.
    kwargs: dict, optional
        Parameters for parent class.
    """

    ATTENTION_TYPE_ADD = "additive"
    ATTENTION_TYPE_MUL = "multiplicative"

    def __init__(
        self,
        units=32,
        attention_width=None,
        attention_type=ATTENTION_TYPE_ADD,
        return_attention=False,
        history_only=False,
        kernel_initializer="glorot_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_additive_bias=True,
        use_attention_bias=True,
        attention_activation=None,
        attention_regularizer_weight=0.0,
        **kwargs,
    ):
        """Layer initialization."""
        super().__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError(
                "No implementation for attention type : " + attention_type
            )

    def get_config(self):
        """Return the config of the layer."""
        config = {
            "units": self.units,
            "attention_width": self.attention_width,
            "attention_type": self.attention_type,
            "return_attention": self.return_attention,
            "history_only": self.history_only,
            "use_additive_bias": self.use_additive_bias,
            "use_attention_bias": self.use_attention_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "attention_activation": keras.activations.serialize(
                self.attention_activation
            ),
            "attention_regularizer_weight": self.attention_regularizer_weight,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Build the layer."""
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super().build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(
            shape=(feature_dim, self.units),
            name=f"{self.name}_Add_Wt",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.Wx = self.add_weight(
            shape=(feature_dim, self.units),
            name=f"{self.name}_Add_Wx",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_additive_bias:
            self.bh = self.add_weight(
                shape=(self.units,),
                name=f"{self.name}_Add_bh",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        self.Wa = self.add_weight(
            shape=(self.units, 1),
            name=f"{self.name}_Add_Wa",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_attention_bias:
            self.ba = self.add_weight(
                shape=(1,),
                name=f"{self.name}_Add_ba",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(
            shape=(feature_dim, feature_dim),
            name=f"{self.name}_Mul_Wa",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_attention_bias:
            self.ba = self.add_weight(
                shape=(1,),
                name=f"{self.name}_Mul_ba",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

    def call(self, inputs, mask=None, **kwargs):
        """Compute the output of the layer."""
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e -= 10000.0 * (
                1.0
                - K.cast(lower <= indices, K.floatx())
                * K.cast(indices < upper, K.floatx())
            )
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            e -= 10000.0 * (
                (1.0 - mask) * (1.0 - K.permute_dimensions(mask, (0, 2, 1)))
            )

        # a_{t} = \text{softmax}(e_t)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        a = e / K.sum(e, axis=-1, keepdims=True)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(
                K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len)
            )
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        """Compute the mask of the layer."""
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return (
            self.attention_regularizer_weight
            * K.sum(
                K.square(
                    K.batch_dot(attention, K.permute_dimensions(attention, (0, 2, 1)))
                    - eye
                )
            )
            / batch_size
        )

    @staticmethod
    def get_custom_objects():
        """Get custom objects for Keras."""
        return {"SeqSelfAttention": SeqSelfAttention}
