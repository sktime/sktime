import numpy as np
from tensorflow import keras

from keras_self_attention import SeqSelfAttention
from .util import TestMaskShape


class TestLoss(TestMaskShape):

    def test_loss(self):
        attention = SeqSelfAttention(return_attention=False,
                                     attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                     kernel_regularizer=keras.regularizers.l2(1e-6),
                                     bias_regularizer=keras.regularizers.l1(1e-6),
                                     attention_regularizer_weight=1e-4,
                                     name='Attention')
        sentences, input_data, token_dict = self.get_input_data()
        model = self.get_model(attention, token_dict)
        sentence_len = input_data.shape[1]
        model.fit(
            x=input_data,
            y=np.zeros((len(sentences), sentence_len, 1)),
            epochs=10,
        )
        self.assertTrue(model is not None)
