import copy

import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from sktime.tests.test_switch import run_test_module_changed

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._torch_self_attention")
    or not _check_soft_dependencies("torch", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)


class TestDeepcopy:
    """Tests that the layer stays copyable once it has been run.

    ``forward`` stores intermediate tensors on the layer. Non-leaf tensors do
    not support the deepcopy protocol, so any such tensor left attached to the
    autograd graph makes every fitted estimator containing this layer
    uncopyable, which breaks ``test_deepcopy_fitted`` for TapNet and CNTC.
    """

    @staticmethod
    def _make(weight=0.0, attention_type=None):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        if attention_type is None:
            attention_type = SeqSelfAttentionTorch.ATTENTION_TYPE_ADD
        return SeqSelfAttentionTorch(
            input_dim=8,
            return_attention=False,
            attention_type=attention_type,
            attention_regularizer_weight=weight,
        )

    @pytest.mark.parametrize("weight", [0.0, 1e-4])
    def test_deepcopy_after_forward(self, weight):
        """Layer is deep-copyable after a forward pass, with and without the
        attention regularizer enabled."""
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        for attention_type in (
            SeqSelfAttentionTorch.ATTENTION_TYPE_ADD,
            SeqSelfAttentionTorch.ATTENTION_TYPE_MUL,
        ):
            attention = self._make(weight, attention_type)
            torch_randn = _safe_import("torch.randn")
            attention(torch_randn(2, 5, 8))

            attention_copy = copy.deepcopy(attention)

            for name, value in vars(attention_copy).items():
                grad_fn = getattr(value, "grad_fn", None)
                assert grad_fn is None, (
                    f"{name} is still attached to the autograd graph on the copy"
                )

    def test_deepcopy_preserves_behaviour(self):
        """The copy is independent of the original and computes the same output."""
        attention = self._make(weight=1e-4)
        torch_randn = _safe_import("torch.randn")
        x = torch_randn(2, 5, 8)
        attention(x)

        attention_copy = copy.deepcopy(attention)

        params = list(attention.parameters())
        params_copy = list(attention_copy.parameters())
        assert len(params) == len(params_copy)
        assert all(p is not q for p, q in zip(params, params_copy))

        torch_allclose = _safe_import("torch.allclose")
        assert torch_allclose(attention(x), attention_copy(x), atol=1e-6)

    def test_regularizer_loss_is_differentiable(self):
        """``attention_regularizer_loss`` must stay attached to the graph.

        It is meant to be added to the caller's training loss. Detaching it to
        make the layer copyable would silently turn the regularizer into a
        no-op, so guard against that regression here.
        """
        attention = self._make(weight=1e-2)
        torch_randn = _safe_import("torch.randn")
        attention(torch_randn(2, 5, 8))

        loss = attention.attention_regularizer_loss
        assert loss is not None
        assert loss.requires_grad
        assert loss.grad_fn is not None

        attention.zero_grad()
        loss.backward()
        grads = [p.grad for p in attention.parameters() if p.grad is not None]
        assert grads, "regularizer produced no gradients"
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_transient_tensors_are_detached(self):
        """``intensity`` and ``attention`` are inspection only, so they are
        detached at assignment rather than at copy time."""
        attention = self._make(weight=1e-4)
        torch_randn = _safe_import("torch.randn")
        attention(torch_randn(2, 5, 8))

        assert attention.intensity is not None
        assert attention.intensity.grad_fn is None
        assert attention.attention is not None
        assert attention.attention.grad_fn is None
