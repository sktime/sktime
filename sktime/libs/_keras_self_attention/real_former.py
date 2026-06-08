"""RealFormer layers."""

from sktime.utils.dependencies import _safe_import

keras = _safe_import("tensorflow.keras")
K = _safe_import("tensorflow.keras.backend")


class ResidualScaledDotProductAttention(keras.layers.Layer):
    r"""Residual scaled dot product attention layer.

    The attention layer that takes three inputs representing queries, keys and values.

    \text{Attention}(Q, K, V, Prev) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}} + Prev) V

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, return_attention=False, history_only=False, **kwargs):
        """Initialize the layer.

        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super().__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = self.attention = None

    def get_config(self):
        """Return the config of the layer."""
        config = {
            "return_attention": self.return_attention,
            "history_only": self.history_only,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        """Compute the mask for the layer."""
        mask = mask[0]
        if self.return_attention:
            mask = [mask, mask[-1], None]
        return [mask, mask[-1]]

    def call(self, inputs, mask=None, **kwargs):
        """Compute the output of the layer."""
        if len(inputs) == 4:
            query, key, value, prev = inputs
            mask = mask[1]
        else:
            query = key = value = inputs[0]
            prev = inputs[1]
            mask = mask[0]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(
            K.cast(feature_dim, dtype=K.floatx())
        )
        new_prev = e = e + prev
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(0, key_len), axis=0)
            upper = K.expand_dims(K.arange(0, query_len), axis=-1)
            e -= 10000.0 * K.expand_dims(K.cast(indices > upper, K.floatx()), axis=0)
        if mask is not None:
            e -= 10000.0 * (1.0 - K.cast(K.expand_dims(mask, axis=-2), K.floatx()))
        self.intensity = e
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        self.attention = e / K.sum(e, axis=-1, keepdims=True)
        v = K.batch_dot(self.attention, value)
        output = [v, new_prev]
        if self.return_attention:
            output.append(self.attention)
        return output
