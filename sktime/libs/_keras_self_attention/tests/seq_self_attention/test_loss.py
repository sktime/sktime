import numpy as np
import pytest

from sktime.libs._keras_self_attention import SeqSelfAttention
from sktime.libs._keras_self_attention.tests.seq_self_attention.util import (
    TestMaskShape,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

keras = _safe_import("tensorflow.keras")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._keras_self_attention")
    or not _check_soft_dependencies("tensorflow", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
class TestLoss(TestMaskShape):
    def test_loss(self):
        attention = SeqSelfAttention(
            return_attention=False,
            attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
            kernel_regularizer=keras.regularizers.l2(1e-6),
            bias_regularizer=keras.regularizers.l1(1e-6),
            attention_regularizer_weight=1e-4,
            name="Attention",
        )
        sentences, input_data, token_dict = self.get_input_data()
        model = self.get_model(attention, token_dict)
        sentence_len = input_data.shape[1]
        model.fit(
            x=input_data,
            y=np.zeros((len(sentences), sentence_len, 1)),
            epochs=10,
        )
        assert model is not None
