import os
import tempfile
import unittest

import numpy as np
from tensorflow import keras

from keras_self_attention import ResidualScaledDotProductAttention


class TestResidualScaledDotProductAttention(unittest.TestCase):

    def test_history(self):
        input_layer = keras.layers.Input(
            shape=(5,),
            name='Input',
        )
        prev_layer = keras.layers.Input(
            shape=(5, 5),
            name='Prev',
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
        att_layer, _, att_weights = ResidualScaledDotProductAttention(
            history_only=True,
            return_attention=True,
            name='Attention',
        )([embed_layer, embed_layer, embed_layer, prev_layer])
        model = keras.models.Model(inputs=[input_layer, prev_layer], outputs=[att_layer, att_weights])
        model.compile(optimizer='adam', loss='mse')
        model_path = os.path.join(tempfile.gettempdir(), 'keras_self_att_test_sl_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'ResidualScaledDotProductAttention': ResidualScaledDotProductAttention,
            },
        )
        inputs = np.array([[1, 2, 3, 1, 0]])
        prev = np.zeros((1, 5, 5))
        predicts = model.predict([inputs, prev])
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

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(5,),
            name='Input',
        )
        prev_layer = keras.layers.Input(
            shape=(5, 5),
            name='Prev',
        )
        embed_layer = keras.layers.Embedding(
            input_dim=4,
            output_dim=5,
            mask_zero=True,
            weights=[
                np.array([
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.2, 0.3, 0.4, 0.6, 0.5],
                    [0.4, 0.7, 0.2, 0.6, 0.9],
                    [0.3, 0.5, 0.8, 0.9, 0.1],
                ]),
            ],
            name='Embedding',
        )(input_layer)
        att_layer, _ = ResidualScaledDotProductAttention(name='Attention')([embed_layer, prev_layer])
        model = keras.models.Model(inputs=[input_layer, prev_layer], outputs=att_layer)
        model.compile(optimizer='adam', loss='mse')
        inputs = np.array([[1, 2, 3, 1, 0]])
        prev = np.zeros((1, 5, 5))
        predict = model.predict([inputs, prev])[0]
        self.assertTrue(np.allclose(predict[0], predict[3]))
        self.assertTrue(np.allclose(
            np.asarray([0.27883747, 0.45767492, 0.47448885, 0.69199574, 0.47368336]),
            predict[2],
        ), predict[2])
