# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for series transformers.

This package forwards leaf modules from the legacy
``sktime.transformations.series.*`` paths to their new flat locations
under ``sktime.transformations``. New code should import from
``sktime.transformations`` directly.

Subpackages still housed under ``series.*`` (``detrend``, ``holiday``,
``kalman_filter``, ``tsfel``) are *not* aliased here; they continue to live
as real subpackages until they are flattened in their own slices.
"""

import importlib
import sys
import warnings

# Leaf modules that were moved from ``series/<name>.py`` to
# ``transformations/<name>.py``. Registering them in ``sys.modules`` makes
# imports such as ``from sktime.transformations.series.boxcox import X`` keep
# working without leaving stub files on disk.
_MOVED_LEAVES = (
    "_clasp_numba",
    "acf",
    "adapt",
    "adi_cv",
    "augmenter",
    "basisfunction",
    "binning",
    "bkfilter",
    "bollinger",
    "boxcox",
    "cffilter",
    "clasp",
    "clear_sky",
    "cos",
    "date",
    "degree_day",
    "difference",
    "dilation_mapping",
    "dobin",
    "dropna",
    "dummies",
    "exponent",
    "fabba",
    "feature_selection",
    "filter",
    "fourier",
    "func_transform",
    "hidalgo",
    "hpfilter",
    "hurst",
    "impute",
    "kinematic",
    "lag",
    "matrix_profile",
    "outlier_detection",
    "paa",
    "peak",
    "sax",
    "scaledasinh",
    "scaledlogit",
    "signature",
    "subsequence_extraction",
    "subset",
    "summarize",
    "temporian",
    "theta",
    "time_since",
    "vmd",
)

for _name in _MOVED_LEAVES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(
        f"sktime.transformations.{_name}"
    )

warnings.warn(
    "sktime.transformations.series is deprecated and will be removed in a "
    "future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.boxcox.BoxCoxTransformer).",
    DeprecationWarning,
    stacklevel=2,
)
