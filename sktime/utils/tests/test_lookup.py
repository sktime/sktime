"""Tests for case-insensitive module lookup utilities."""

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils._lookup"]),
    reason="Run if relevant content has changed.",
)
def test_lowercase_importer_maps_module_names():
    """Names are indexed by their lower-case spelling."""
    from sktime.utils._lookup import _lowercase_importer

    name_map = _lowercase_importer("numpy")

    assert name_map["scalartype"] == "ScalarType"
    assert name_map["ndarray"] == "ndarray"


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils._lookup"]),
    reason="Run if relevant content has changed.",
)
def test_lowercase_importer_missing_module_raises():
    """A missing module path raises ModuleNotFoundError."""
    from sktime.utils._lookup import _lowercase_importer

    with pytest.raises(ModuleNotFoundError, match="not_a_real_module_xyz"):
        _lowercase_importer("not_a_real_module_xyz")


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils._lookup"]),
    reason="Run if relevant content has changed.",
)
@pytest.mark.parametrize("alias", ["ScalarType", "scalartype", "SCALARTYPE"])
def test_lookup_is_case_insensitive(alias):
    """Objects are resolved from a module by case-insensitive name."""
    import numpy as np

    from sktime.utils._lookup import _lookup

    assert _lookup(alias, "numpy") is np.ScalarType


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils._lookup"]),
    reason="Run if relevant content has changed.",
)
def test_lookup_uses_optional_alias_dict():
    """Optional alias_dict maps extra aliases onto canonical names."""
    import numpy as np

    from sktime.utils._lookup import _lookup

    obj = _lookup("st", "numpy", alias_dict={"st": "ScalarType"})

    assert obj is np.ScalarType


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils._lookup"]),
    reason="Run if relevant content has changed.",
)
def test_lookup_alias_dict_does_not_clobber_exported_name():
    """A bad alias_dict value must not replace a name exported by the module."""
    import numpy as np

    from sktime.utils._lookup import _lookup

    obj = _lookup("ndarray", "numpy", alias_dict={"ndarray": "not_a_real_name_xyz"})

    assert obj is np.ndarray


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils._lookup"]),
    reason="Run if relevant content has changed.",
)
def test_lookup_alias_dict_ignores_unknown_target():
    """alias_dict entries whose target is not an exported name are ignored."""
    from sktime.utils._lookup import _lookup

    with pytest.raises(ValueError, match="st"):
        _lookup("st", "numpy", alias_dict={"st": "not_a_real_name_xyz"})


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils._lookup"]),
    reason="Run if relevant content has changed.",
)
def test_lookup_unknown_alias_raises():
    """Unknown names raise ValueError after the module map is checked."""
    from sktime.utils._lookup import _lookup

    with pytest.raises(ValueError, match="not_a_real_name_xyz"):
        _lookup("not_a_real_name_xyz", "numpy")


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(["sktime.utils._lookup"]),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("alias", ["AdamW", "adamw", "ADAMW"])
def test_lookup_resolves_torch_optim_names(alias):
    """torch.optim classes are resolved case-insensitively by name."""
    import torch

    from sktime.utils._lookup import _lookup

    assert _lookup(alias, "torch.optim") is torch.optim.AdamW


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none")
    or not run_test_module_changed(["sktime.utils._lookup"]),
    reason="skip test if required soft dependency not available",
)
def test_lookup_resolves_monkeypatched_torch_optim_name(monkeypatch):
    """Names present on the live torch.optim module resolve even if not shipped."""
    import torch

    from sktime.utils._lookup import _lookup

    class ExtraOptimizer(torch.optim.SGD):
        """Optimizer name that is not a real torch.optim export."""

    monkeypatch.setattr(torch.optim, "ExtraOptimizer", ExtraOptimizer, raising=False)

    assert _lookup("ExtraOptimizer", "torch.optim") is ExtraOptimizer
    assert _lookup("extraoptimizer", "torch.optim") is ExtraOptimizer
