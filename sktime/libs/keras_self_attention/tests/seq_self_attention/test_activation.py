from tensorflow import keras

from keras_self_attention import SeqSelfAttention
from .util import TestMaskShape


class TestActivation(TestMaskShape):

    def test_attention_activation(self):
        attention = SeqSelfAttention(return_attention=True,
                                     attention_width=3,
                                     kernel_regularizer=keras.regularizers.l2(1e-4),
                                     bias_regularizer=keras.regularizers.l1(1e-4),
                                     attention_activation='sigmoid',
                                     name='Attention')
        self.check_mask_shape(attention)
