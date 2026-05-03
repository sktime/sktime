# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for signature-based transformers.

This package forwards to the new top-level subpackage
``sktime.transformations.signature_based``.
"""

import importlib
import sys
import warnings

from sktime.transformations.signature_based import SignatureTransformer  # noqa: F401

__all__ = [
    "SignatureTransformer",
]

# Make legacy submodule paths resolve to the new top-level signature_based
# subpackage. All were private files; aliased only for pickle compat.
_legacy_to_new = {
    "_augmentations": "sktime.transformations.signature_based._augmentations",
    "_checks": "sktime.transformations.signature_based._checks",
    "_compute": "sktime.transformations.signature_based._compute",
    "_rescaling": "sktime.transformations.signature_based._rescaling",
    "_signature_method": "sktime.transformations.signature_based._signature_method",
    "_window": "sktime.transformations.signature_based._window",
    "SignatureTransformer": "sktime.transformations.signature_based._signature_method",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.panel.signature_based is deprecated and will be "
    "removed in a future release. Import from "
    "sktime.transformations.signature_based instead.",
    DeprecationWarning,
    stacklevel=2,
)
