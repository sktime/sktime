"""Deprecated. Please import from ``sktime.detection`` instead.

This package is kept for backward compatibility and will be removed in a
future release. All segment anomaly detectors have been natively merged into
``sktime.detection``.
"""

from sktime.detection.skchange_aseg.mvcapa import MVCAPA  # noqa: F401

__all__ = [
    "CAPA",
    "CircularBinarySegmentation",
    "MVCAPA",
    "StatThresholdAnomaliser",
]


def __getattr__(name):
    import importlib
    import warnings

    _deprecated = {
        "CAPA": "sktime.detection._capa",
        "CircularBinarySegmentation": "sktime.detection._circular_binseg",
        "StatThresholdAnomaliser": "sktime.detection._stat_threshold_anomaliser",
    }
    if name in _deprecated:
        warnings.warn(
            f"Importing {name} from sktime.detection.skchange_aseg is deprecated. "
            "Please import directly from sktime.detection instead. "
            "This legacy path will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(_deprecated[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
