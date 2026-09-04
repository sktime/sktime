import pytest
from skbase.utils.dependencies import _safe_import

torch = _safe_import("torch")
if torch is None:
    pytest.skip(
        "torch is not installed; skipping torch utils tests", allow_module_level=True
    )


def test_clone_state_dict_roundtrip():
    """clone_state_dict should produce a state dict that can be loaded into
    a fresh module with identical parameters."""
    import torch.nn as nn

    from sktime.utils.torch_utils import clone_state_dict, load_state_dict_into

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    for param in m.parameters():
        param.data.uniform_(-1.0, 1.0)

    sd = clone_state_dict(m)

    m2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    load_state_dict_into(m2, sd)

    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(p1.detach().cpu(), p2.detach().cpu())
