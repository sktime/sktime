import pytest
from skbase.utils.dependencies import _safe_import

torch = _safe_import("torch")
if torch is None:
    pytest.skip(
        "torch is not installed; skipping early stopper tests", allow_module_level=True
    )


def test_early_stopper_saves_and_restores_state_dict():
    """_EarlyStopper.early_stop should store a cloned state_dict
    (not a deepcopy of the module) and that state_dict can later
    be loaded into a fresh module to reproduce parameters.
    """
    import torch.nn as nn

    from sktime.forecasting.cinn import _EarlyStopper
    from sktime.utils.torch_utils import load_state_dict_into

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
    for p in model.parameters():
        p.data.uniform_(-1.0, 1.0)

    orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    es = _EarlyStopper(patience=1, min_delta=0)

    es.early_stop(validation_loss=0.1, model=model)

    assert getattr(es, "_best_state_dict", None) is not None

    for p in model.parameters():
        p.data.add_(1.0)

    fresh = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
    load_state_dict_into(fresh, es._best_state_dict)

    for name, tensor in fresh.state_dict().items():
        assert torch.allclose(tensor.detach().cpu(), orig_state[name])
