"""Tests for the pytorch deep learning adapter."""

import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.tests.test_switch import run_test_module_changed

KINDS = ["classifier", "regressor"]

CASES = {
    "classifier": {
        "criterion": "nllloss",
        "criterion_extra": {"activation": "softmax"},
    },
    "regressor": {"criterion": "l1loss", "criterion_extra": {}},
}

run_test = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(
        [
            "sktime.base.adapters._pytorch",
            "sktime.classification.deep_learning",
            "sktime.regression.deep_learning",
        ]
    ),
    reason="skip test if required soft dependency not available",
)


def _make(kind, **kwargs):
    params = {
        "num_epochs": 2,
        "batch_size": 4,
        "callbacks": None,
        "hidden_dim": 5,
        "n_layers": 1,
        "dropout": 0.0,
        "random_state": 42,
    }
    params.update(kwargs)
    if kind == "classifier":
        from sktime.classification.deep_learning.mlp import MLPClassifierTorch

        return MLPClassifierTorch(**params)
    from sktime.regression.deep_learning.mlp import MLPRegressorTorch

    return MLPRegressorTorch(**params)


def _data(kind):
    if kind == "classifier":
        from sktime.datasets import load_unit_test

        return load_unit_test(split="train")
    from sktime.utils._testing.panel import make_regression_problem

    return make_regression_problem(n_instances=10, n_timepoints=12)


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_base_classes_share_the_adapter(kind):
    """Test that both base classes take the training machinery from the adapter."""
    from sktime.base.adapters._pytorch import _PytorchDeepAdapter

    est = _make(kind)
    assert isinstance(est, _PytorchDeepAdapter)
    for method in ["_instantiate_optimizer", "_instantiate_criterion", "_run_epoch"]:
        assert getattr(type(est), method) is getattr(_PytorchDeepAdapter, method)


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_optimizer_str_looked_up_in_torch_optim(kind):
    """Test that any optimizer in torch.optim can be selected by its name."""
    import torch

    X, y = _data(kind)

    for optimizer in ["sgd", "SGD", "SgD"]:
        est = _make(kind, optimizer=optimizer).fit(X, y)
        assert isinstance(est._optimizer, torch.optim.SGD)

    if hasattr(torch.optim, "Adafactor"):
        est = _make(kind, optimizer="adafactor").fit(X, y)
        assert isinstance(est._optimizer, torch.optim.Adafactor)

    for optimizer in ["not_an_optimizer", "", "optimizer", "lr_scheduler"]:
        with pytest.raises(ValueError, match="Unknown optimizer"):
            _make(kind, optimizer=optimizer).fit(X, y)


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_optimizer_class(kind):
    """Test that an optimizer passed as a class is instantiated on the network."""
    import torch

    X, y = _data(kind)

    est = _make(
        kind, optimizer=torch.optim.SGD, lr=0.02, optimizer_kwargs={"momentum": 0.9}
    ).fit(X, y)

    assert isinstance(est._optimizer, torch.optim.SGD)
    assert est._optimizer.param_groups[0]["lr"] == 0.02
    assert est._optimizer.param_groups[0]["momentum"] == 0.9
    bound = [p for g in est._optimizer.param_groups for p in g["params"]]
    assert all(a is b for a, b in zip(bound, est.network.parameters()))


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_optimizer_instance_is_bound_to_network(kind):
    """Test that an optimizer passed as an instance updates the network.

    Failure case of bug #10990: the network is only built in ``fit``, so an
    optimizer instance is bound to other parameters than the ones of the network,
    and the network was never updated.
    """
    import torch

    X, y = _data(kind)

    placeholder = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(placeholder.parameters(), lr=0.01)

    est_instance = _make(kind, optimizer=optimizer).fit(X, y)

    bound = [p for g in est_instance._optimizer.param_groups for p in g["params"]]
    network_params = list(est_instance.network.parameters())
    assert len(bound) == len(network_params)
    assert all(a is b for a, b in zip(bound, network_params))

    est_str = _make(kind, optimizer="adam", lr=0.01).fit(X, y)

    y_instance = est_instance.predict(X)
    y_str = est_str.predict(X)
    if np.issubdtype(np.asarray(y_instance).dtype, np.number):
        np.testing.assert_allclose(y_instance, y_str, rtol=1e-6)
    else:
        np.testing.assert_array_equal(y_instance, y_str)


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_optimizer_instance_hyperparameters(kind):
    """Test that hyperparameters of an optimizer instance are carried over."""
    import torch

    X, y = _data(kind)

    placeholder = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        placeholder.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001
    )

    est = _make(kind, optimizer=optimizer).fit(X, y)
    assert isinstance(est._optimizer, torch.optim.SGD)
    assert est._optimizer.param_groups[0]["lr"] == 0.05
    assert est._optimizer.param_groups[0]["momentum"] == 0.9
    assert est._optimizer.param_groups[0]["weight_decay"] == 0.0001

    est = _make(kind, optimizer=optimizer, lr=0.02).fit(X, y)
    assert est._optimizer.param_groups[0]["lr"] == 0.02
    assert est._optimizer.param_groups[0]["momentum"] == 0.9

    est = _make(kind, optimizer=optimizer, optimizer_kwargs={"momentum": 0.5}).fit(X, y)
    assert est._optimizer.param_groups[0]["momentum"] == 0.5


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_optimizer_kwargs(kind):
    """Test that optimizer_kwargs are passed to the optimizer."""
    X, y = _data(kind)

    est = _make(
        kind,
        optimizer="SGD",
        optimizer_kwargs={"momentum": 0.9, "weight_decay": 0.0001},
    ).fit(X, y)

    assert est._optimizer.param_groups[0]["momentum"] == 0.9
    assert est._optimizer.param_groups[0]["weight_decay"] == 0.0001


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_optimizer_invalid_raises(kind):
    """Test that an invalid optimizer raises an informative error."""
    import torch

    X, y = _data(kind)

    for optimizer in [42, 0, torch.nn.Linear, torch.nn.Linear(4, 3)]:
        with pytest.raises(TypeError, match="optimizer"):
            _make(kind, optimizer=optimizer).fit(X, y)


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_criterion_str_looked_up_in_torch_nn(kind):
    """Test that any loss function in torch.nn can be selected by its name."""
    import torch

    X, y = _data(kind)
    case = CASES[kind]
    expected = getattr(
        torch.nn, {"nllloss": "NLLLoss", "l1loss": "L1Loss"}[case["criterion"]]
    )

    for criterion in [case["criterion"], case["criterion"].upper()]:
        est = _make(kind, criterion=criterion, **case["criterion_extra"]).fit(X, y)
        assert isinstance(est._criterion, expected)

    for criterion in ["not_a_loss", "Linear"]:
        with pytest.raises(ValueError, match="Unknown criterion"):
            _make(kind, criterion=criterion).fit(X, y)


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_activation_str_looked_up_in_torch_nn(kind):
    """Test that any activation in torch.nn can be selected by its name."""
    import torch

    case = CASES[kind]

    for activation in ["softmax", "Softmax", "SOFTMAX"]:
        est = _make(kind, activation=activation, criterion=case["criterion"])
        assert isinstance(est._callable_activations["activation"], torch.nn.Softmax)

    with pytest.raises(ValueError, match="not a valid PyTorch activation"):
        _make(kind, activation="not_an_activation", criterion=case["criterion"])


@run_test
@pytest.mark.parametrize("kind", KINDS)
def test_callbacks_str_looked_up_in_lr_scheduler(kind):
    """Test that any scheduler in torch.optim.lr_scheduler can be selected by name."""
    import torch

    X, y = _data(kind)

    for callbacks in ["constantlr", "ConstantLR", "CONSTANTLR"]:
        est = _make(kind, callbacks=callbacks).fit(X, y)
        assert isinstance(est._schedulers[0], torch.optim.lr_scheduler.ConstantLR)

    est = _make(kind, callbacks=("constantlr", "ConstantLR")).fit(X, y)
    assert len(est._schedulers) == 2

    for callbacks in ["not_a_scheduler", "LRScheduler"]:
        with pytest.raises(ValueError, match="Unknown learning rate scheduler"):
            _make(kind, callbacks=callbacks).fit(X, y)
