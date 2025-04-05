from tensorflow import keras

from keras_self_attention import SeqSelfAttention
from .util import TestMaskShape


class TestLocal(TestMaskShape):

    def check_local_range(self, attention_type):
        attention = SeqSelfAttention(return_attention=True,
                                     attention_width=5,
                                     attention_type=attention_type,
                                     kernel_regularizer=keras.regularizers.l2(1e-4),
                                     bias_regularizer=keras.regularizers.l1(1e-4),
                                     name='Attention')
        self.check_mask_shape(attention)

    def test_add(self):
        self.check_local_range(attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD)

    def test_mul(self):
        self.check_local_range(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)
