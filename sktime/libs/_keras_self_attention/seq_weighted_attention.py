"""Sequential weighted attention layer."""

from sktime.utils.dependencies import _safe_import

keras = _safe_import("tensorflow.keras")
K = _safe_import("tensorflow.keras.backend")


class SeqWeightedAttention(keras.layers.Layer):
    r"""Sequential weighted attention layer.

    Y = \text{softmax}(XW + b) X

    See: https://arxiv.org/pdf/1708.00524.pdf
    """

    def __init__(self, use_bias=True, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.return_attention = return_attention
        self.W, self.b = None, None

    def get_config(self):
        """Return the config of the layer."""
        config = {
            "use_bias": self.use_bias,
            "return_attention": self.return_attention,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Create the weights of the layer."""
        self.W = self.add_weight(
            shape=(int(input_shape[2]), 1),
            name=f"{self.name}_W",
            initializer=keras.initializers.get("uniform"),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(1,),
                name=f"{self.name}_b",
                initializer=keras.initializers.get("zeros"),
            )
        super().build(input_shape)

    def call(self, x, mask=None):
        """Compute the output of the layer."""
        logits = K.dot(x, self.W)
        if self.use_bias:
            logits += self.b
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            logits -= 10000.0 * (1.0 - mask)
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return input_shape[0], output_len

    def compute_mask(self, _, input_mask=None):
        """Compute the mask for the layer."""
        if self.return_attention:
            return [None, None]
        return None

    @staticmethod
    def get_custom_objects():
        """Return the custom objects of the layer."""
        return {"SeqWeightedAttention": SeqWeightedAttention}
