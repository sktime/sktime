import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon


def _make_simple_panel():
    """Two simple series, 4 time points each."""
    idx = pd.MultiIndex.from_product(
        [["s1", "s2"], pd.period_range("2000-01", periods=4, freq="M")],
        names=["instance", "time"],
    )
    y = pd.Series([1, 2, 3, 4, 10, 20, 30, 40], index=idx, name="y")
    return y


def _loop_baseline_last(y, fh):
    """Baseline: loop over instances and use single-series NaiveForecaster."""
    loop_parts = []
    for inst in ["s1", "s2"]:
        y_inst = y.xs(inst, level="instance")
        f_inst = NaiveForecaster(strategy="last")
        f_inst.fit(y_inst)
        y_pred_inst = f_inst.predict(fh)
        y_pred_inst.index = pd.MultiIndex.from_product(
            [[inst], y_pred_inst.index],
            names=["instance", "time"],
        )
        loop_parts.append(y_pred_inst)

    y_loop = pd.concat(loop_parts).sort_index()
    return y_loop


def test_is_multiseries_y_detects_multiindex_series():
    """_is_multiseries_y should be True for MultiIndex Series (instance, time)."""
    y = _make_simple_panel()
    f = NaiveForecaster(strategy="last")

    # manually set internal y to avoid datatype checks in fit
    f._y = y

    assert f._is_multiseries_y()


def test_multiseries_vectorized_matches_loop_last():
    """Vectorized multiseries path should match loop-over-instances baseline."""
    y = _make_simple_panel()
    fh = ForecastingHorizon([1, 2], is_relative=True)

    # 1. loop baseline (single-series NaiveForecaster per instance)
    y_loop = _loop_baseline_last(y, fh)

    # 2. vectorized multiseries via internal helper
    f = NaiveForecaster(strategy="last")
    f._y = y
    # cutoff = last observed time index across panel
    f._cutoff = y.index.get_level_values("time").max()

    y_vec = f._predict_multiseries_vectorized(fh=fh)

    # align indices just in case
    y_vec = y_vec.sort_index()
    y_loop = y_loop.sort_index()

    assert np.allclose(y_loop.values, y_vec.values)
