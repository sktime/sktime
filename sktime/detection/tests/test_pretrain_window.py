import numpy as np
import pandas as pd

from sktime.detection.naive import NaivePretrainWindowDetector


def _make_single_series(seed=42):
    """Single series with spikes at positions 25-30 and 65-70."""
    n = 100
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    values[:25] = 1
    values[25:30] = 10.0
    values[30:65] = 1
    values[65:70] = 20.0
    return pd.Series(values), pd.DataFrame({"ilocs": [25, 65]})


def _make_panel_series(seed=42):
    """Same data as _make_single_series but as panel with pd-multiindex.

    Returns two instances with the same spike pattern.
    """
    X_single, y_single = _make_single_series(seed=seed)
    n = len(X_single)

    idx = pd.MultiIndex.from_arrays(
        [[0] * n + [1] * n, list(range(n)) + list(range(n))],
        names=["instances", "timepoints"],
    )
    X = pd.DataFrame({"value": np.tile(X_single.values, 2)}, index=idx)
    y = y_single  # same event positions apply per instance
    return X, y


# todo: test cases
# - test that anomaly idx - window_length < 0 is handled correctly
# - panel data is handled correctly, multiindexes, etc.!
def test_naive_pretrain_window_detector():
    # X, y = _make_single_series()
    X, y = _make_panel_series()
    npwd = NaivePretrainWindowDetector(window_length=5)
    breakpoint()
    npwd.pretrain(X, y)

