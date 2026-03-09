import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._torch_self_attention")
    or not _check_soft_dependencies("torch", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)


class TestValidation:
    def test_output_shape(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(return_attention=False, input_dim=8)
        torch_randn = _safe_import("torch.randn")
        x = torch_randn(3, 7, 8)
        y = attention(x)
        assert y.shape == x.shape

    def test_input_dim_mismatch(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(return_attention=False, input_dim=8)
        torch_randn = _safe_import("torch.randn")
        x = torch_randn(2, 5, 9)
        with pytest.raises(ValueError):
            attention(x)

    def test_invalid_input_shape(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(return_attention=False, input_dim=8)
        torch_randn = _safe_import("torch.randn")
        x = torch_randn(2, 8)
        with pytest.raises(ValueError):
            attention(x)
