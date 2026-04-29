# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Asserts that legacy ``series.detrend`` import paths still resolve."""

import importlib
import sys
import warnings


def _reset_legacy_modules():
    """Drop cached legacy modules so the shim's warning fires again."""
    for name in list(sys.modules):
        if name.startswith("sktime.transformations.series.detrend"):
            del sys.modules[name]


def test_legacy_package_import_works_and_warns():
    _reset_legacy_modules()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module("sktime.transformations.series.detrend")
    assert any(
        issubclass(warning.category, DeprecationWarning)
        and "sktime.transformations.series.detrend" in str(warning.message)
        for warning in w
    )


def test_legacy_public_classes_are_identical():
    from sktime.transformations.deseasonalize import (
        ConditionalDeseasonalizer,
        Deseasonalizer,
        STLTransformer,
    )
    from sktime.transformations.detrend import Detrender
    from sktime.transformations.mstl import MSTL
    from sktime.transformations.series.detrend import (
        ConditionalDeseasonalizer as Cd2,
        Deseasonalizer as Dz2,
        Detrender as D2,
        MSTL as M2,
        STLTransformer as S2,
    )
    assert ConditionalDeseasonalizer is Cd2
    assert Deseasonalizer is Dz2
    assert Detrender is D2
    assert MSTL is M2
    assert STLTransformer is S2


def test_legacy_public_submodule_import_mstl():
    # ``mstl`` was a public submodule of series.detrend; this path is part of
    # the back-compat contract. The shim registers it dynamically in
    # sys.modules, so pyright cannot resolve it statically.
    from sktime.transformations.mstl import MSTL as M_new
    from sktime.transformations.series.detrend.mstl import (  # pyright: ignore[reportMissingImports]
        MSTL as M_old,
    )
    assert M_new is M_old
