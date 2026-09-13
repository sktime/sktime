"""Tests for case-insensitive module lookup utilities."""

import pytest

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
def test_lookup_unknown_alias_raises():
    """Unknown names raise ValueError after the module map is checked."""
    from sktime.utils._lookup import _lookup

    with pytest.raises(ValueError, match="not_a_real_name_xyz"):
        _lookup("not_a_real_name_xyz", "numpy")
