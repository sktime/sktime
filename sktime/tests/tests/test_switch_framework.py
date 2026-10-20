"""Regression tests for core framework change selection."""

import pytest

from sktime.base import BaseObject
from sktime.tests.test_switch import _run_test_for_class, run_test_for_class


@pytest.mark.parametrize(
    "module", ["registry", "datatypes", "utils", "forecasting", None]
)
@pytest.mark.parametrize("case", ["core", "noncore", "missing", "vm"])
def test_framework_changes(monkeypatch, module, case):
    """Framework changes select core objects without overriding skip conditions."""
    from skbase.utils import dependencies

    from sktime.tests import _config
    from sktime.utils import git_diff

    core, deps, vm = case != "noncore", case != "missing", case == "vm"

    class TestObject(BaseObject):
        _tags = {"tests:core": core, "tests:vm": vm}

    monkeypatch.setattr(_config, "ONLY_CHANGED_MODULES", True)
    monkeypatch.setattr(_config, "ONLY_VM_ESTIMATORS", False)
    monkeypatch.setattr(dependencies, "_check_estimator_deps", lambda *a, **k: deps)
    monkeypatch.setattr(git_diff, "get_packages_with_changed_specs", lambda: [])
    monkeypatch.setattr(git_diff, "is_class_changed", lambda cls: False)
    monkeypatch.setattr(
        git_diff, "is_module_changed", lambda name: name == f"sktime.{module}"
    )
    expected = core and deps and not vm and module in ("registry", "datatypes", "utils")
    reason = "True_changed_framework" if expected else "False_no_change"
    if not deps or vm:
        reason = "False_required_deps_missing" if not deps else "False_requires_vm"
    _run_test_for_class.cache_clear()
    try:
        assert run_test_for_class(TestObject, return_reason=True) == (expected, reason)
        assert run_test_for_class(TestObject) is expected
    finally:
        _run_test_for_class.cache_clear()
