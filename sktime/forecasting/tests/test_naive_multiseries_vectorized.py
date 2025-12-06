import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon


def _make_simple_panel():
    idx = pd.MultiIndex.from_product(
        [["s1", "s2"], pd.period_range("2000-01", periods=4, freq="M")],
        names=["instance", "time"],
    )
    y = pd.Series([1, 2, 3, 4, 10, 20, 30, 40], index=idx)
    return y


def _predict_multiseries_vectorized_demo(y, fh, strategy="last"):
    """Prototype of a vectorized panel naive forecast (for discussion)."""
    from sktime.forecasting.base import ForecastingHorizon

    cutoff = y.index.get_level_values("time").max()

    idx = y.index
    inst_level = idx.names[0] if idx.names[0] is not None else 0
    time_level = idx.names[1] if idx.names[1] is not None else 1

    if strategy == "last":
        last_vals = (
            y.groupby(level=inst_level)
            .tail(1)
            .droplevel(time_level)
        )
    elif strategy == "mean":
        last_vals = y.groupby(level=inst_level).mean()
    else:
        raise ValueError("Only 'last' and 'mean' supported in this prototype.")

    if not isinstance(fh, ForecastingHorizon):
        fh = ForecastingHorizon(fh, is_relative=True)

    time_index = fh.to_absolute(cutoff).to_pandas()

    inst_index = last_vals.index
    multi_idx = pd.MultiIndex.from_product(
        [inst_index, time_index],
        names=[inst_level, time_level],
    )

    n_horizons = len(time_index)
    vals = np.repeat(last_vals.to_numpy(), n_horizons)

    return pd.Series(vals, index=multi_idx, name=y.name)


def test_multiseries_last_matches_loop_prototype():
    y = _make_simple_panel()
    fh = ForecastingHorizon([1, 2], is_relative=True)

    # loop baseline
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

    # vectorized demo
    y_vec = _predict_multiseries_vectorized_demo(y, fh, strategy="last").sort_index()

    assert np.allclose(y_loop.values, y_vec.values)
