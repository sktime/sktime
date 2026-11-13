"""Forecast-frame correctness and deterministic performance regressions (#11329)."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import _reduce, make_reduction


def _reference_create_fcst_df(target_date, origin_df, fill=None):
    """Construct a frame through the original per-instance horizon machinery."""
    if isinstance(target_date, ForecastingHorizon):
        fh = target_date.to_absolute(cutoff=None)
    else:
        fh = ForecastingHorizon(pd.Index(target_date), is_relative=False)
    index = fh.get_expected_pred_idx(origin_df)
    columns = (
        [origin_df.name] if isinstance(origin_df, pd.Series) else origin_df.columns
    )
    result = pd.DataFrame(
        0 if fill is None else fill, index=index, columns=columns, dtype="float64"
    )
    if isinstance(origin_df, pd.Series) and not isinstance(index, pd.MultiIndex):
        result = result.iloc[:, 0].rename(origin_df.name)
    return result


def _history(index_kind, layout, series=False):
    """Make sparse, unsorted instances with different history lengths and cutoffs."""
    if index_kind == "integer":
        times = pd.Index(range(50))
    elif index_kind == "period":
        times = pd.period_range("2024-01", periods=50, freq="M")
    else:
        timezone = "America/New_York" if index_kind == "timezone" else None
        times = pd.date_range("2024-03-01", periods=50, freq="D", tz=timezone)

    if layout == "flat":
        index = times[:20].rename("time")
    else:
        keys = [("west", "b"), ("east", "a"), ("west", "c")]
        names = ["region", "id", "time"] if layout == "hierarchical" else ["id", "time"]
        index = pd.MultiIndex.from_tuples(
            [
                (key if layout == "hierarchical" else (key[-1],)) + (t,)
                for key, stop in zip(keys, [20, 25, 15])
                for t in times[:stop]
            ],
            names=names,
        )
    values = np.arange(len(index), dtype=float)
    origin = pd.Series(values, index=index, name="y")
    return (origin if series else origin.to_frame()), times[[45, 42, 44]]


@pytest.mark.parametrize("index_kind", ["integer", "period", "datetime", "timezone"])
@pytest.mark.parametrize("layout", ["flat", "panel", "hierarchical"])
@pytest.mark.parametrize("as_horizon", [False, True])
@pytest.mark.parametrize("series", [False, True])
def test_create_fcst_df_matches_reference(index_kind, layout, as_horizon, series):
    """Preserve absolute horizons, sorting, row order, names, types, and fill."""
    origin, dates = _history(index_kind, layout, series=series)
    target = ForecastingHorizon(dates, is_relative=False) if as_horizon else dates
    count = 1 if layout == "flat" else 3
    fill = np.arange(count * len(dates), dtype=float).reshape(-1, 1)
    expected = _reference_create_fcst_df(target, origin, fill)
    actual = _reduce._create_fcst_df(target, origin, fill)
    if isinstance(expected, pd.Series):
        assert_series_equal(actual, expected)
    else:
        assert_frame_equal(actual, expected)


@pytest.mark.parametrize("layout", ["flat", "panel", "hierarchical"])
@pytest.mark.parametrize("dates", [[], [30, 30], [30.5], ["invalid"]])
def test_create_fcst_df_validates_horizon(layout, dates):
    """The fast path must not accept invalid horizons that the old path rejects."""
    origin, _ = _history("integer", layout)
    with pytest.raises((TypeError, ValueError)) as reference_error:
        _reference_create_fcst_df(dates, origin)
    with pytest.raises(type(reference_error.value)):
        _reduce._create_fcst_df(dates, origin)


@pytest.mark.parametrize("layout", ["flat", "panel", "hierarchical"])
def test_create_fcst_df_relative_horizon_requires_cutoff(layout):
    """Do not silently treat relative offsets as absolute dates."""
    origin, _ = _history("integer", layout)
    fh = ForecastingHorizon([1, 3], is_relative=True)
    with pytest.raises((TypeError, ValueError)) as reference_error:
        _reference_create_fcst_df(fh, origin)
    with pytest.raises(type(reference_error.value)):
        _reduce._create_fcst_df(fh, origin)


@pytest.mark.parametrize("named", [False, True])
def test_create_fcst_df_multiple_columns(named):
    """Multivariate fill stays aligned with non-rectangular hierarchy keys."""
    origin, dates = _history("integer", "hierarchical")
    origin["second"] = -origin["y"]
    if not named:
        origin.index = origin.index.set_names([None] * origin.index.nlevels)
    fill = np.arange(18, dtype=float).reshape(9, 2)
    assert_frame_equal(
        _reduce._create_fcst_df(dates, origin, fill),
        _reference_create_fcst_df(dates, origin, fill),
    )


@pytest.mark.parametrize("layout", ["flat", "panel", "hierarchical"])
def test_create_fcst_df_rejects_incompatible_frequency(layout):
    """Keep the frequency validation performed by the original horizon path."""
    origin, _ = _history("period", layout)
    dates = pd.period_range("2028-01-01", periods=3, freq="D")
    with pytest.raises(ValueError, match="Frequencies"):
        _reference_create_fcst_df(dates, origin)
    with pytest.raises(ValueError, match="Frequencies"):
        _reduce._create_fcst_df(dates, origin)


@pytest.mark.parametrize("layout", ["panel", "hierarchical"])
def test_create_fcst_df_bypasses_per_instance_horizon_checks(monkeypatch, layout):
    """Absolute panel dates must not invoke per-instance cutoff/type checks."""
    origin, dates = _history("integer", layout)
    expected = _reference_create_fcst_df(dates, origin)

    def unexpected_horizon_lookup(*args, **kwargs):
        raise AssertionError("Absolute panel dates do not need per-instance cutoffs")

    monkeypatch.setattr(
        ForecastingHorizon, "get_expected_pred_idx", unexpected_horizon_lookup
    )
    assert_frame_equal(_reduce._create_fcst_df(dates, origin), expected)


@pytest.mark.parametrize("layout", ["panel", "hierarchical"])
@pytest.mark.parametrize("index_kind", ["integer", "period", "datetime", "timezone"])
def test_recursive_panel_predictions_match_reference(monkeypatch, layout, index_kind):
    """A pooled recursive model must retain actual forecasts, not only their shape."""
    y, _ = _history(index_kind, layout)
    # This PR optimizes frame construction; per-instance cutoff semantics are
    # addressed separately in #11746. Use a shared cutoff for this regression.
    y = y.groupby(level=list(range(y.index.nlevels - 1)), sort=False).head(15)
    forecaster = make_reduction(
        LinearRegression(), strategy="recursive", window_length=4, pooling="global"
    ).fit(y)
    prediction = forecaster.predict(fh=[1, 3, 5])
    monkeypatch.setattr(_reduce, "_create_fcst_df", _reference_create_fcst_df)
    expected = forecaster.predict(fh=[1, 3, 5])
    assert_frame_equal(prediction, expected, check_exact=True)
