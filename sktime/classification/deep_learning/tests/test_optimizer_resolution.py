"""Tests for optimizer string resolution in ``BaseDeepClassifierPytorch``."""

import pytest

torch = pytest.importorskip("torch")

from sktime.classification.deep_learning.mlp import MLPClassifier  # noqa: E402
from sktime.utils.dependencies import _safe_import  # noqa: E402


@pytest.fixture
def optimizer_base_class():
    """The ``torch.optim.Optimizer`` base class used to validate candidates."""
    return _safe_import("torch.optim.Optimizer")


@pytest.fixture
def bare_clf():
    """An unfitted classifier; ``_resolve_optimizer`` needs no fitted state."""
    return MLPClassifier()


def test_resolve_optimizer_curated_alias(bare_clf, optimizer_base_class):
    """Lower-case aliases resolve via the curated ``_all_optimizers`` map."""
    assert (
        bare_clf._resolve_optimizer("adamw", optimizer_base_class)
        is torch.optim.AdamW
    )


def test_resolve_optimizer_case_insensitive(bare_clf, optimizer_base_class):
    """Arbitrary case works, e.g. ``"RMSprop"`` rather than ``"rmsprop"``."""
    assert (
        bare_clf._resolve_optimizer("RMSprop", optimizer_base_class)
        is torch.optim.RMSprop
    )


def test_resolve_optimizer_falls_back_to_torch_optim(
    bare_clf, optimizer_base_class, monkeypatch
):
    """Names not in the curated map are looked up directly in ``torch.optim``."""
    class Muon(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3):
            super().__init__(params, {"lr": lr})

        def step(self, closure=None):  # pragma: no cover - not called
            raise NotImplementedError

    monkeypatch.setattr(torch.optim, "Muon", Muon, raising=False)
    assert bare_clf._resolve_optimizer("Muon", optimizer_base_class) is Muon


def test_resolve_optimizer_unknown_raises(bare_clf, optimizer_base_class):
    """Unknown names raise a ``ValueError`` with a helpful message."""
    with pytest.raises(ValueError, match="Unknown optimizer"):
        bare_clf._resolve_optimizer("NotAnOptimizer", optimizer_base_class)
