# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for tsfel transformers.

This package forwards to the flat layout. New code should import from
``sktime.transformations.tsfel`` directly.
"""

import importlib
import sys
import warnings

from sktime.transformations.tsfel import TSFELTransformer  # noqa: F401

__all__ = [
    "TSFELTransformer",
]

# Make legacy submodule path resolve to the new flat module.
#   - ``_tsfel`` was a private file; aliased only for pickle compat.
sys.modules[f"{__name__}._tsfel"] = importlib.import_module(
    "sktime.transformations.tsfel"
)

warnings.warn(
    "sktime.transformations.series.tsfel is deprecated and will be removed "
    "in a future release. Import from sktime.transformations.tsfel instead.",
    DeprecationWarning,
    stacklevel=2,
)
