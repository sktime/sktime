import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._torch_self_attention")
    or not _check_soft_dependencies("torch", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)


class TestLoss:
    def test_loss(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(
            return_attention=False,
            attention_type=SeqSelfAttentionTorch.ATTENTION_TYPE_MUL,
            attention_regularizer_weight=1e-4,
        )
        torch_randn = _safe_import("torch.randn")
        x = torch_randn(4, 6, 8)
        _ = attention(x)
        loss = attention.attention_regularizer_loss
        assert loss is not None
        assert loss.item() >= 0.0
