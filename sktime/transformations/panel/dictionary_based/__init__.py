# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for dictionary-based transformers.

This package forwards to the flat layout. New code should import from
``sktime.transformations`` directly (e.g.
``from sktime.transformations.sfa import SFA``).
"""

import importlib
import sys
import warnings

from sktime.transformations.paa_legacy import PAAlegacy  # noqa: F401
from sktime.transformations.sax_legacy import SAXlegacy  # noqa: F401
from sktime.transformations.sfa import SFA  # noqa: F401
from sktime.transformations.sfa_fast import SFAFast  # noqa: F401

__all__ = ["SFA", "SFAFast", "PAAlegacy", "SAXlegacy"]

# Make legacy submodule paths resolve to the new flat modules.
# All were private files; aliased only for pickle compat.
_legacy_to_new = {
    "_paa": "sktime.transformations.paa_legacy",
    "_sax": "sktime.transformations.sax_legacy",
    "_sfa": "sktime.transformations.sfa",
    "_sfa_fast": "sktime.transformations.sfa_fast",
    "_sfa_numba": "sktime.transformations._sfa_numba",
    "_sfa_fast_numba": "sktime.transformations._sfa_fast_numba",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.panel.dictionary_based is deprecated and will be "
    "removed in a future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.sfa.SFA).",
    DeprecationWarning,
    stacklevel=2,
)
