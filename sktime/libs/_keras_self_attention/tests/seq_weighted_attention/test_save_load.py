import os
import tempfile

import numpy as np
import pytest

from sktime.libs._keras_self_attention import SeqWeightedAttention as Attention
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

keras = _safe_import("tensorflow.keras")


@pytest.mark.xfail(reason="Unknown failure reason - TODO investigate")
@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._keras_self_attention")
    or not _check_soft_dependencies("tensorflow", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
class TestSaveLoad:
    def _test_save_load(self, attention):
        inputs = keras.layers.Input(shape=(None,), name="Input")
        embd = keras.layers.Embedding(
            input_dim=3, output_dim=5, mask_zero=True, name="Embedding"
        )(inputs)
        lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(units=7, return_sequences=True), name="Bi-LSTM"
        )(embd)
        if attention.return_attention:
            layer, weights = attention(lstm)
        else:
            layer = attention(lstm)
        dense = keras.layers.Dense(units=2, activation="softmax", name="Softmax")(layer)
        loss = {"Softmax": "sparse_categorical_crossentropy"}
        if attention.return_attention:
            outputs = [dense, weights]
            loss[attention.name] = "mse"
        else:
            outputs = dense
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss=loss)
        model_path = os.path.join(
            tempfile.gettempdir(),
            "keras_weighted_att_test_sl_%f.h5" % np.random.random(),
        )
        model.save(model_path)
        model = keras.models.load_model(
            model_path, custom_objects=Attention.get_custom_objects()
        )
        model.summary(line_length=100)
        if attention.return_attention:
            assert len(model.outputs) == 2
        else:
            assert len(model.outputs) == 1

    def test_default(self):
        self._test_save_load(Attention(name="Attention"))

    def test_return_attention(self):
        self._test_save_load(
            Attention(return_attention=True, use_bias=False, name="Attention")
        )
