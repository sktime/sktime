# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Asserts that legacy ``series.tsfel`` import paths still resolve."""

import importlib
import sys
import warnings


def _reset_legacy_modules():
    """Drop cached legacy modules so the shim's warning fires again."""
    for name in list(sys.modules):
        if name.startswith("sktime.transformations.series.tsfel"):
            del sys.modules[name]


def test_legacy_package_import_works_and_warns():
    _reset_legacy_modules()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module("sktime.transformations.series.tsfel")
    assert any(
        issubclass(warning.category, DeprecationWarning)
        and "sktime.transformations.series.tsfel" in str(warning.message)
        for warning in w
    )


def test_legacy_public_class_is_identical():
    from sktime.transformations.series.tsfel import TSFELTransformer as T2
    from sktime.transformations.tsfel import TSFELTransformer
    assert TSFELTransformer is T2
