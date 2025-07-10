import numpy as np
import pytest

from sktime.libs._keras_self_attention import ScaledDotProductAttention
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

keras = _safe_import("tensorflow.keras")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._keras_self_attention")
    or not _check_soft_dependencies("tensorflow", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
class TestAttention:
    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(5,),
            name="Input",
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
        att_layer = ScaledDotProductAttention(name="Attention")(embed_layer)
        model = keras.models.Model(inputs=input_layer, outputs=att_layer)
        model.compile(optimizer="adam", loss="mse")
        model.summary()
        inputs = np.array([[1, 2, 3, 1, 0]])
        predict = model.predict(inputs)[0]
        assert np.allclose(predict[0], predict[3])
        assert np.allclose(
            np.asarray([0.27883747, 0.45767492, 0.47448885, 0.69199574, 0.47368336]),
            predict[2],
        ), predict[2]
