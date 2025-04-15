import os
import tempfile

import numpy as np
import pytest

from sktime.libs._keras_self_attention import SeqSelfAttention
from sktime.libs._keras_self_attention.tests.seq_self_attention.util import (
    TestMaskShape,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

keras = _safe_import("tensorflow.keras")


@pytest.mark.xfail(reason="Unknown failure reason - TODO investigate")
@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._keras_self_attention")
    or not _check_soft_dependencies("tensorflow", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
class TestSaveLoad(TestMaskShape):
    def test_save_load(self):
        _, _, token_dict = self.get_input_data()
        model = self.get_model(SeqSelfAttention(name="Attention"), token_dict)
        model_path = os.path.join(
            tempfile.gettempdir(),
            "keras_self_att_test_save_load_%f.h5" % np.random.random(),
        )
        model.save(model_path)
        model = keras.models.load_model(
            model_path, custom_objects={"SeqSelfAttention": SeqSelfAttention}
        )
        model.summary()
        assert model is not None

    def test_save_load_with_loss(self):
        attention = SeqSelfAttention(
            return_attention=True,
            attention_width=7,
            attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_regularizer=keras.regularizers.l1(1e-4),
            attention_regularizer_weight=1e-3,
            name="Attention",
        )
        _, _, token_dict = self.get_input_data()
        model = self.get_model(attention, token_dict)
        model_path = os.path.join(
            tempfile.gettempdir(),
            "keras_self_att_test_sl_with_loss_%f.h5" % np.random.random(),
        )
        model.save(model_path)
        model = keras.models.load_model(
            model_path, custom_objects=SeqSelfAttention.get_custom_objects()
        )
        model.summary()
        assert model is not None
