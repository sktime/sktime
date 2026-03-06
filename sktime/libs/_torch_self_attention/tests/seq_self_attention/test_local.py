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


class TestLocal(TestMaskShapeTorch):
    def check_local_range(self, attention_type):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(
            return_attention=True,
            attention_width=5,
            attention_type=attention_type,
        )
        self.check_mask_shape(attention)

    def test_add(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        self.check_local_range(attention_type=SeqSelfAttentionTorch.ATTENTION_TYPE_ADD)

    def test_mul(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        self.check_local_range(attention_type=SeqSelfAttentionTorch.ATTENTION_TYPE_MUL)
