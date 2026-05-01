# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for shapelet transformers.

This package forwards to the flat layout. New code should import from
``sktime.transformations`` directly (e.g.
``from sktime.transformations.shapelet_transform import ShapeletTransform``).
"""

import importlib
import sys
import warnings

from sktime.transformations.shapelet_transform import (  # noqa: F401
    RandomShapeletTransform,
    ShapeletTransform,
)
from sktime.transformations.shapelet_transform_pyts import (  # noqa: F401
    ShapeletTransformPyts,
)

__all__ = [
    "ShapeletTransform",
    "RandomShapeletTransform",
    "ShapeletTransformPyts",
]

# Make legacy submodule paths resolve to the new flat modules.
# Both were private files; aliased only for pickle compat.
_legacy_to_new = {
    "_shapelet_transform": "sktime.transformations.shapelet_transform",
    "_shapelet_transform_pyts": "sktime.transformations.shapelet_transform_pyts",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.panel.shapelet_transform is deprecated and will "
    "be removed in a future release. Import from sktime.transformations "
    "directly (e.g. sktime.transformations.shapelet_transform.ShapeletTransform).",
    DeprecationWarning,
    stacklevel=2,
)
