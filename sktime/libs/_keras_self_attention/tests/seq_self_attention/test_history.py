import pytest

from sktime.libs._keras_self_attention import SeqSelfAttention
from sktime.libs._keras_self_attention.tests.seq_self_attention.util import (
    TestMaskShape,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._keras_self_attention")
    or not _check_soft_dependencies("tensorflow", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
class TestHistory(TestMaskShape):
    def test_history(self):
        attention = SeqSelfAttention(
            return_attention=True,
            attention_width=3,
            history_only=True,
            name="Attention",
        )
        self.check_mask_shape(attention)

    def test_infinite_history(self):
        attention = SeqSelfAttention(
            return_attention=True, history_only=True, name="Attention"
        )
        self.check_mask_shape(attention)
