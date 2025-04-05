import unittest

import numpy as np
from tensorflow import keras

from keras_self_attention import ScaledDotProductAttention


class TestHistory(unittest.TestCase):

    def test_history(self):
        input_layer = keras.layers.Input(
            shape=(5,),
            name='Input',
        )
        embed_layer = keras.layers.Embedding(
            input_dim=4,
            output_dim=5,
            mask_zero=True,
            weights=[
                np.asarray([
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.2, 0.3, 0.4, 0.6, 0.5],
                    [0.4, 0.7, 0.2, 0.6, 0.9],
                    [0.3, 0.5, 0.8, 0.9, 0.1],
                ]),
            ],
            name='Embedding',
        )(input_layer)
        att_layer, att_weights = ScaledDotProductAttention(
            history_only=True,
            return_attention=True,
            name='Attention',
        )([embed_layer, embed_layer, embed_layer])
        model = keras.models.Model(inputs=input_layer, outputs=[att_layer, att_weights])
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        inputs = np.array([[1, 2, 3, 1, 0]])
        predicts = model.predict(inputs)
        results, weights = predicts[0][0], predicts[1][0]
        self.assertFalse(np.allclose(results[0], results[3]))
        self.assertTrue(np.allclose(
            np.asarray([0.2, 0.3, 0.4, 0.6, 0.5]),
            results[0],
        ), results[0])
        for i in range(4):
            for j in range(5):
                if j > i:
                    self.assertEqual(0.0, weights[i][j])
                else:
                    self.assertLess(0.0, weights[i][j])
