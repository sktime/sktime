from sktime.libs.keras_self_attention import SeqSelfAttention
from sktime.libs.keras_self_attention.tests.seq_self_attention.util import TestMaskShape
from sktime.utils.dependencies import _safe_import

keras = _safe_import("tensorflow.keras")


class TestMul(TestMaskShape):

    def test_multiplicative(self):
        attention = SeqSelfAttention(
            return_attention=True,
            attention_width=15,
            attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_regularizer=keras.regularizers.l1(1e-4),
            name="Attention",
        )
        self.check_mask_shape(attention)

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            SeqSelfAttention(return_attention=True, attention_type="random")
