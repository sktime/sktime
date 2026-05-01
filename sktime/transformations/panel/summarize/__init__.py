# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for panel summarize transformers.

This package forwards to the flat layout. The four classes formerly in
``panel.summarize._extract`` now live in
``sktime.transformations.interval_features``. New code should import from
there directly.
"""

import importlib
import sys
import warnings

from sktime.transformations.interval_features import (  # noqa: F401
    DerivativeSlopeTransformer,
    FittedParamExtractor,
    PlateauFinder,
    RandomIntervalFeatureExtractor,
)

__all__ = [
    "DerivativeSlopeTransformer",
    "PlateauFinder",
    "RandomIntervalFeatureExtractor",
    "FittedParamExtractor",
]

# Make legacy submodule path resolve to the new flat module.
# ``_extract`` was a private file; aliased only for pickle compat.
sys.modules[f"{__name__}._extract"] = importlib.import_module(
    "sktime.transformations.interval_features"
)

warnings.warn(
    "sktime.transformations.panel.summarize is deprecated and will be removed "
    "in a future release. Import from sktime.transformations.interval_features "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)
