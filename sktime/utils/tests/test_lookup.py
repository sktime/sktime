"""Testing the case insensitive class lookup utilities."""

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._lookup import _lc_class_dict, _lookup_class

# the lookup utilities are generic, torch.optim is used as a test module,
# since it contains classes, a base class of those, and sub-modules
TORCH_OPTIM = "torch.optim"
TORCH_OPTIMIZER = "torch.optim.Optimizer"


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils._lookup"]),
    reason="Run if the lookup module has changed.",
)
def test_lc_class_dict_missing_module():
    """Test that the lookup of a module that cannot be imported is empty."""
    assert dict(_lc_class_dict("nonexistent_module_for_testing")) == {}


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(["sktime.utils._lookup"]),
    reason="skip test if required soft dependency not available",
)
def test_lc_class_dict_collects_subclasses():
    """Test that all strict subclasses in the module are collected, and only those."""
    import torch.optim

    lookup = _lc_class_dict(TORCH_OPTIM, TORCH_OPTIMIZER)

    expected = {
        name.lower(): obj
        for name, obj in vars(torch.optim).items()
        if isinstance(obj, type)
        and issubclass(obj, torch.optim.Optimizer)
        and obj is not torch.optim.Optimizer
    }
    assert dict(lookup) == expected

    # the optimizers that were previously available through a curated dictionary
    assert {"adam", "adamw", "sgd", "rmsprop", "lbfgs"}.issubset(lookup)

    # the base class itself, and objects that are not classes, are not collected
    assert "optimizer" not in lookup
    assert "lr_scheduler" not in lookup


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(["sktime.utils._lookup"]),
    reason="skip test if required soft dependency not available",
)
def test_lc_class_dict_without_base_class():
    """Test that all public classes are collected if no base class is passed."""
    import torch.optim

    lookup = _lc_class_dict(TORCH_OPTIM)

    assert lookup["optimizer"] is torch.optim.Optimizer
    assert lookup["adam"] is torch.optim.Adam


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(["sktime.utils._lookup"]),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("name", ["adamw", "AdamW", "ADAMW", "AdAmW"])
def test_lookup_class_is_case_insensitive(name):
    """Test that a class is looked up in any capitalization of its name."""
    import torch.optim

    assert _lookup_class(name, TORCH_OPTIM, TORCH_OPTIMIZER) is torch.optim.AdamW


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(["sktime.utils._lookup"]),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("name", ["not_a_class", "", "optimizer", "lr_scheduler"])
def test_lookup_class_not_found(name):
    """Test that None is returned if the module contains no such class.

    ``optimizer`` and ``lr_scheduler`` are in ``torch.optim``, but are the base
    class and a sub-module, so they are not valid results of the lookup.
    """
    assert _lookup_class(name, TORCH_OPTIM, TORCH_OPTIMIZER) is None


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(["sktime.utils._lookup"]),
    reason="skip test if required soft dependency not available",
)
def test_lookup_class_alias_dict():
    """Test that aliases are resolved before the lookup, case insensitively."""
    import torch.optim

    alias_dict = {"adaptive_moments": "Adam"}

    for name in ["adaptive_moments", "Adaptive_Moments"]:
        found = _lookup_class(name, TORCH_OPTIM, TORCH_OPTIMIZER, alias_dict)
        assert found is torch.optim.Adam

    # names that are not aliases are still looked up directly
    found = _lookup_class("sgd", TORCH_OPTIM, TORCH_OPTIMIZER, alias_dict)
    assert found is torch.optim.SGD


# the lookup locations used by the deep learning estimators, as
# (module, base class, names that must be found, names that must not be found)
TORCH_LOOKUPS = [
    (
        "torch.optim",
        "torch.optim.Optimizer",
        ["adam", "sgd", "rmsprop"],
        ["optimizer", "lr_scheduler"],
    ),
    (
        "torch.nn",
        "torch.nn.modules.loss._Loss",
        ["mseloss", "crossentropyloss", "nllloss"],
        ["linear", "module"],
    ),
    (
        "torch.nn",
        "torch.nn.Module",
        ["relu", "logsoftmax", "adaptivelogsoftmaxwithloss"],
        ["module", "functional"],
    ),
    (
        "torch.optim.lr_scheduler",
        "torch.optim.lr_scheduler.LRScheduler",
        ["steplr", "reducelronplateau", "constantlr"],
        ["lrscheduler", "_lrscheduler"],
    ),
]


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(["sktime.utils._lookup"]),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("module_path,base_class_path,present,absent", TORCH_LOOKUPS)
def test_lookups_used_by_deep_learning_estimators(
    module_path, base_class_path, present, absent
):
    """Test the lookup locations that the deep learning estimators look names up in."""
    lookup = _lc_class_dict(module_path, base_class_path)

    for name in present:
        assert name in lookup
        assert _lookup_class(name, module_path, base_class_path) is lookup[name]

    # base classes and objects that are not classes are not in the lookup
    for name in absent:
        assert name not in lookup
        assert _lookup_class(name, module_path, base_class_path) is None
