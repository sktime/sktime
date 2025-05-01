import os
import tempfile

import numpy as np
import pytest

from sktime.libs._keras_self_attention import ScaledDotProductAttention
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
    def test_save_load(self):
        input_q = keras.layers.Input(shape=(5, 3), name="Input-Q")
        input_k = keras.layers.Input(shape=(4, 3), name="Input-K")
        input_v = keras.layers.Input(shape=(4, 6), name="Input-V")
        attention, weights = ScaledDotProductAttention(
            return_attention=True,
            history_only=True,
            name="Attention",
        )([input_q, input_k, input_v])
        model = keras.models.Model(
            inputs=[input_q, input_k, input_v], outputs=[attention, weights]
        )
        model.compile(optimizer="adam", loss="mse")
        model_path = os.path.join(
            tempfile.gettempdir(), "keras_self_att_test_sl_%f.h5" % np.random.random()
        )
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={
                "ScaledDotProductAttention": ScaledDotProductAttention,
            },
        )
        model.summary(line_length=120)
        assert model is not None
