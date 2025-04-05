from sktime.libs.keras_self_attention import SeqSelfAttention
from sktime.libs.keras_self_attention.tests.seq_self_attention.util import TestMaskShape
from sktime.utils.dependencies import _safe_import

keras = _safe_import("tensorflow.keras")


class TestMask(TestMaskShape):

    def test_return_attention(self):
        attention = SeqSelfAttention(return_attention=True,
                                     kernel_regularizer=keras.regularizers.l2(1e-4),
                                     bias_regularizer=keras.regularizers.l1(1e-4),
                                     name='Attention')
        self.check_mask_shape(attention)
