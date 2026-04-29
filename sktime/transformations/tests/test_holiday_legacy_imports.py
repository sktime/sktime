# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Asserts that legacy ``series.holiday`` import paths still resolve."""

import importlib
import sys
import warnings


def _reset_legacy_modules():
    """Drop cached legacy modules so the shim's warning fires again."""
    for name in list(sys.modules):
        if name.startswith("sktime.transformations.series.holiday"):
            del sys.modules[name]


def test_legacy_package_import_works_and_warns():
    _reset_legacy_modules()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module("sktime.transformations.series.holiday")
    assert any(
        issubclass(warning.category, DeprecationWarning)
        and "sktime.transformations.series.holiday" in str(warning.message)
        for warning in w
    )


def test_legacy_public_classes_are_identical():
    from sktime.transformations.country_holidays import CountryHolidaysTransformer
    from sktime.transformations.financial_holidays import (
        FinancialHolidaysTransformer,
    )
    from sktime.transformations.holidayfeats import HolidayFeatures
    from sktime.transformations.series.holiday import (
        CountryHolidaysTransformer as C2,
        FinancialHolidaysTransformer as F2,
        HolidayFeatures as H2,
    )
    assert CountryHolidaysTransformer is C2
    assert FinancialHolidaysTransformer is F2
    assert HolidayFeatures is H2


def test_legacy_public_submodule_imports():
    # ``country_holidays`` and ``financial_holidays`` were public submodules;
    # this is part of the back-compat contract. The shim registers them
    # dynamically in sys.modules, so pyright cannot see them statically.
    from sktime.transformations.country_holidays import (
        CountryHolidaysTransformer as C_new,
    )
    from sktime.transformations.financial_holidays import (
        FinancialHolidaysTransformer as F_new,
    )
    from sktime.transformations.series.holiday.country_holidays import (  # pyright: ignore[reportMissingImports]
        CountryHolidaysTransformer as C_old,
    )
    from sktime.transformations.series.holiday.financial_holidays import (  # pyright: ignore[reportMissingImports]
        FinancialHolidaysTransformer as F_old,
    )
    assert C_new is C_old
    assert F_new is F_old
