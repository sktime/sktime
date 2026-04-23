import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._torch_self_attention")
    or not _check_soft_dependencies("torch", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)


class TestSaveLoad:
    def test_state_dict_roundtrip(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(
            return_attention=False,
            attention_type=SeqSelfAttentionTorch.ATTENTION_TYPE_ADD,
            input_dim=8,
        )
        torch_randn = _safe_import("torch.randn")
        x = torch_randn(2, 5, 8)
        y1 = attention(x)
        state = attention.state_dict()

        attention2 = SeqSelfAttentionTorch(
            return_attention=False,
            attention_type=SeqSelfAttentionTorch.ATTENTION_TYPE_ADD,
            input_dim=8,
        )
        attention2.load_state_dict(state)
        y2 = attention2(x)
        torch_allclose = _safe_import("torch.allclose")
        assert torch_allclose(y1, y2, atol=1e-6)
