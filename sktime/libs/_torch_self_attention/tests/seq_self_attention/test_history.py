import pytest

from sktime.libs._torch_self_attention.tests.seq_self_attention.util import (
    TestMaskShapeTorch,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._torch_self_attention")
    or not _check_soft_dependencies("torch", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)


class TestHistory(TestMaskShapeTorch):
    def test_history(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(
            return_attention=True,
            attention_width=3,
            history_only=True,
        )
        self.check_mask_shape(attention)

    def test_infinite_history(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(return_attention=True, history_only=True)
        self.check_mask_shape(attention)
