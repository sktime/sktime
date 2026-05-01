# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for panel transformers.

This package forwards leaf modules from the legacy
``sktime.transformations.panel.*`` paths to their new flat locations
under ``sktime.transformations``. New code should import from
``sktime.transformations`` directly.

Subpackages (``dictionary_based``, ``rocket``, ``shapelet_transform``,
``signature_based``, ``summarize``) are aliased via their own legacy
``__init__.py`` shims; this top-level shim aliases the leaf modules only.
"""

import importlib
import sys
import warnings

# Leaf modules moved from ``panel/<name>.py`` to ``transformations/<name>.py``
# (or, for the two collisions, to a renamed top-level file).
_LEAF_RENAMES = {
    # Same-name moves
    "_catch22_numba": "_catch22_numba",
    "_shapelet_transform_numba": "_shapelet_transform_numba",
    "catch22": "catch22",
    "catch22wrapper": "catch22wrapper",
    "channel_selection": "channel_selection",
    "compose_distance": "compose_distance",
    "dwt": "dwt",
    "hog1d": "hog1d",
    "interpolate": "interpolate",
    "padder": "padder",
    "pca": "pca",
    "random_intervals": "random_intervals",
    "reduce": "reduce",
    "segment": "segment",
    "slope": "slope",
    "supervised_intervals": "supervised_intervals",
    "truncation": "truncation",
    "tsfeatures": "tsfeatures",
    "tsfresh": "tsfresh",
    # Collision renames
    "matrix_profile": "matrix_profile_panel",
    "compose": "column_concatenator",
}
for _legacy, _new in _LEAF_RENAMES.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(
        f"sktime.transformations.{_new}"
    )

warnings.warn(
    "sktime.transformations.panel is deprecated and will be removed in a "
    "future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.catch22.Catch22).",
    DeprecationWarning,
    stacklevel=2,
)
