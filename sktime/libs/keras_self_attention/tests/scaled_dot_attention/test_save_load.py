import unittest
import os
import tempfile

import numpy as np
from tensorflow import keras

from keras_self_attention import ScaledDotProductAttention


class TestSaveLoad(unittest.TestCase):

    def test_save_load(self):
        input_q = keras.layers.Input(shape=(5, 3), name='Input-Q')
        input_k = keras.layers.Input(shape=(4, 3), name='Input-K')
        input_v = keras.layers.Input(shape=(4, 6), name='Input-V')
        attention, weights = ScaledDotProductAttention(
            return_attention=True,
            history_only=True,
            name='Attention',
        )([input_q, input_k, input_v])
        model = keras.models.Model(inputs=[input_q, input_k, input_v], outputs=[attention, weights])
        model.compile(optimizer='adam', loss='mse')
        model_path = os.path.join(tempfile.gettempdir(), 'keras_self_att_test_sl_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'ScaledDotProductAttention': ScaledDotProductAttention,
            },
        )
        model.summary(line_length=120)
        self.assertTrue(model is not None)
