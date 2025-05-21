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
class TestMask(TestMaskShape):
    def test_return_attention(self):
        attention = SeqSelfAttention(
            return_attention=True,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_regularizer=keras.regularizers.l1(1e-4),
            name="Attention",
        )
        self.check_mask_shape(attention)
