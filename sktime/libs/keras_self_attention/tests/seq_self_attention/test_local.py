from sktime.libs.keras_self_attention import SeqSelfAttention
from sktime.libs.keras_self_attention.tests.seq_self_attention.util import TestMaskShape
from sktime.utils.dependencies import _safe_import

keras = _safe_import("tensorflow.keras")


class TestLocal(TestMaskShape):

    def check_local_range(self, attention_type):
        attention = SeqSelfAttention(
            return_attention=True,
            attention_width=5,
            attention_type=attention_type,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_regularizer=keras.regularizers.l1(1e-4),
            name="Attention",
        )
        self.check_mask_shape(attention)

    def test_add(self):
        self.check_local_range(attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD)

    def test_mul(self):
        self.check_local_range(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)
