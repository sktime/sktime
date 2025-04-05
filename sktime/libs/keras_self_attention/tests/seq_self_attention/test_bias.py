from tensorflow import keras

from keras_self_attention import SeqSelfAttention
from .util import TestMaskShape


class TestBias(TestMaskShape):

    def test_no_bias(self):
        attention = SeqSelfAttention(return_attention=True,
                                     attention_width=3,
                                     kernel_regularizer=keras.regularizers.l2(1e-4),
                                     bias_regularizer=keras.regularizers.l1(1e-4),
                                     use_additive_bias=False,
                                     use_attention_bias=False,
                                     attention_activation='relu',
                                     name='Attention')
        self.check_mask_shape(attention)
