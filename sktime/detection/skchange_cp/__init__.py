"""Deprecated. Please import from ``sktime.detection`` instead.

This package is kept for backward compatibility and will be removed in a
future release. All change point detectors have been natively merged into
``sktime.detection``.
"""

__all__ = [
    "MovingWindow",
    "PELT",
    "SeededBinarySegmentation",
]


def __getattr__(name):
    import importlib
    import warnings

    _deprecated = {
        "PELT": "sktime.detection._pelt",
        "MovingWindow": "sktime.detection._moving_window",
        "SeededBinarySegmentation": "sktime.detection._seeded_binseg",
    }
    if name in _deprecated:
        warnings.warn(
            f"Importing {name} from sktime.detection.skchange_cp is deprecated. "
            "Please import directly from sktime.detection instead. "
            "This legacy path will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(_deprecated[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
