"""Regression tests for local and CI VM estimator selection."""

import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.tests import _config
from sktime.tests.test_switch import _run_test_for_class, run_test_for_class


class _VMForecaster(NaiveForecaster):
    _tags = {"tests:vm": True}


@pytest.mark.parametrize(
    "exclude_vm,only_vm,cls,expected",
    [
        (False, False, _VMForecaster, (True, "True_run_always")),
        (False, False, NaiveForecaster, (True, "True_run_always")),
        (True, False, _VMForecaster, (False, "False_requires_vm")),
        (True, False, NaiveForecaster, (True, "True_run_always")),
        (True, True, _VMForecaster, (True, "True_run_always")),
        (True, True, NaiveForecaster, (False, "False_requires_vm")),
    ],
)
def test_vm_selection(monkeypatch, exclude_vm, only_vm, cls, expected):
    """Local checks include VM estimators; CI retains explicit partitions."""
    monkeypatch.setattr(_config, "ONLY_CHANGED_MODULES", False)
    monkeypatch.setattr(_config, "ONLY_VM_ESTIMATORS", only_vm)
    monkeypatch.setattr(_config, "EXCLUDE_VM_ESTIMATORS", exclude_vm, raising=False)
    _run_test_for_class.cache_clear()
    assert run_test_for_class(cls, return_reason=True) == expected
    assert run_test_for_class(cls()) == expected[0]
    assert run_test_for_class([cls, cls]) == expected[0]


def test_vm_selection_preserves_skip_all(monkeypatch):
    """Allowing local VM tests must not override an estimator's explicit skip."""

    class SkippedVMForecaster(_VMForecaster):
        _tags = {"tests:skip_all": True}

    monkeypatch.setattr(_config, "ONLY_CHANGED_MODULES", False)
    monkeypatch.setattr(_config, "ONLY_VM_ESTIMATORS", False)
    monkeypatch.setattr(_config, "EXCLUDE_VM_ESTIMATORS", False, raising=False)
    assert run_test_for_class(SkippedVMForecaster, return_reason=True) == (
        False,
        "False_exclude_list",
    )
