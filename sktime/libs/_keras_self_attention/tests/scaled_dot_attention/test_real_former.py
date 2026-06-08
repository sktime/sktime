import os
import tempfile

import numpy as np
import pytest

from sktime.libs._keras_self_attention import ResidualScaledDotProductAttention
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

keras = _safe_import("tensorflow.keras")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._keras_self_attention")
    or not _check_soft_dependencies("tensorflow", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
class TestResidualScaledDotProductAttention:
    @pytest.mark.xfail(reason="Unknown failure reason - TODO investigate")
    def test_history(self):
        input_layer = keras.layers.Input(
            shape=(5,),
            name="Input",
        )
        prev_layer = keras.layers.Input(
            shape=(5, 5),
            name="Prev",
        )
        embed_layer = keras.layers.Embedding(
            input_dim=4,
            output_dim=5,
            mask_zero=True,
            weights=[
                np.asarray(
                    [
                        [0.1, 0.2, 0.3, 0.4, 0.5],
                        [0.2, 0.3, 0.4, 0.6, 0.5],
                        [0.4, 0.7, 0.2, 0.6, 0.9],
                        [0.3, 0.5, 0.8, 0.9, 0.1],
                    ]
                ),
            ],
            name="Embedding",
        )(input_layer)
        att_layer, _, att_weights = ResidualScaledDotProductAttention(
            history_only=True,
            return_attention=True,
            name="Attention",
        )([embed_layer, embed_layer, embed_layer, prev_layer])
        model = keras.models.Model(
            inputs=[input_layer, prev_layer], outputs=[att_layer, att_weights]
        )
        model.compile(optimizer="adam", loss="mse")
        model_path = os.path.join(
            tempfile.gettempdir(), "keras_self_att_test_sl_%f.h5" % np.random.random()
        )
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={
                "ResidualScaledDotProductAttention": ResidualScaledDotProductAttention,
            },
        )
        inputs = np.array([[1, 2, 3, 1, 0]])
        prev = np.zeros((1, 5, 5))
        predicts = model.predict([inputs, prev])
        results, weights = predicts[0][0], predicts[1][0]
        assert not np.allclose(results[0], results[3])
        assert not np.allclose(
            np.asarray([0.2, 0.3, 0.4, 0.6, 0.5]),
            results[0],
        ), results[0]
        for i in range(4):
            for j in range(5):
                if j > i:
                    assert weights[i][j] == 0.0
                else:
                    assert 0.0 < weights[i][j] < 1.0

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(5,),
            name="Input",
        )
        prev_layer = keras.layers.Input(
            shape=(5, 5),
            name="Prev",
        )
        embed_layer = keras.layers.Embedding(
            input_dim=4,
            output_dim=5,
            mask_zero=True,
            weights=[
                np.array(
                    [
                        [0.1, 0.2, 0.3, 0.4, 0.5],
                        [0.2, 0.3, 0.4, 0.6, 0.5],
                        [0.4, 0.7, 0.2, 0.6, 0.9],
                        [0.3, 0.5, 0.8, 0.9, 0.1],
                    ]
                ),
            ],
            name="Embedding",
        )(input_layer)
        att_layer, _ = ResidualScaledDotProductAttention(name="Attention")(
            [embed_layer, prev_layer]
        )
        model = keras.models.Model(inputs=[input_layer, prev_layer], outputs=att_layer)
        model.compile(optimizer="adam", loss="mse")
        inputs = np.array([[1, 2, 3, 1, 0]])
        prev = np.zeros((1, 5, 5))
        predict = model.predict([inputs, prev])[0]
        assert np.allclose(predict[0], predict[3])
        assert np.allclose(
            np.asarray([0.27883747, 0.45767492, 0.47448885, 0.69199574, 0.47368336]),
            predict[2],
        ), predict[2]
