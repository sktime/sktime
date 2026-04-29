# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for holiday transformers.

This package forwards to the flat layout. New code should import from
``sktime.transformations`` directly (e.g.
``from sktime.transformations.country_holidays import CountryHolidaysTransformer``).
"""

import importlib
import sys
import warnings

from sktime.transformations.country_holidays import (  # noqa: F401
    CountryHolidaysTransformer,
)
from sktime.transformations.financial_holidays import (  # noqa: F401
    FinancialHolidaysTransformer,
)
from sktime.transformations.holidayfeats import HolidayFeatures  # noqa: F401

__all__ = [
    "CountryHolidaysTransformer",
    "FinancialHolidaysTransformer",
    "HolidayFeatures",
]

# Make legacy submodule paths resolve to the new flat modules.
#   - ``country_holidays`` and ``financial_holidays`` were public submodules.
#   - ``_holidayfeats`` was private; aliased only for pickle compat (objects
#     pickled with previous releases store the old private ``__module__``).
_legacy_to_new = {
    "country_holidays": "sktime.transformations.country_holidays",
    "financial_holidays": "sktime.transformations.financial_holidays",
    "_holidayfeats": "sktime.transformations.holidayfeats",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.series.holiday is deprecated and will be removed "
    "in a future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.country_holidays.CountryHolidaysTransformer).",
    DeprecationWarning,
    stacklevel=2,
)
