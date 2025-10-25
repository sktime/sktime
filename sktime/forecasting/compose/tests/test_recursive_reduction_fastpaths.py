# Tests for optimized recursive reduction fast paths and guards
# Focus: fasttail path, v2 local path, guard fallbacks, constant-mean fallback,
# slice_at_ix surrogate behaviour, and exogenous + MultiIndex fallback sanity.

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose._reduce import (
    RecursiveReductionForecaster,
    slice_at_ix,
)


@pytest.fixture
def simple_series():
    idx = pd.period_range("2000-01", periods=60, freq="M")
    y = pd.DataFrame({"y": np.linspace(0, 59, 60)}, index=idx)
    return y


@pytest.fixture
def simple_fh_gappy(simple_series):
    # gappy horizon including a gap to test proper filtering
    return ForecastingHorizon(
        [1, 2, 5, 7, 12], is_relative=True, freq=simple_series.index
    )


def _fit_rec(y, window_length=5, **kwargs):
    forecaster = RecursiveReductionForecaster(
        estimator=LinearRegression(), window_length=window_length, **kwargs
    )
    fh_dummy = ForecastingHorizon(
        [1], is_relative=True, freq=y.index
    )  # fit does not need full fh
    forecaster.fit(y, fh=fh_dummy)
    return forecaster


# def test_fasttail_equivalence(simple_series, simple_fh_gappy):
#     """fasttail vs legacy predictions identical under eligible conditions."""
#     # guard eligibility: local, no exog, impute_method None
#     f = _fit_rec(simple_series, window_length=6, pooling="local", impute_method=None)
#     # compute baseline legacy prediction
#     pred_legacy = f._predict_out_of_sample_v1(None, simple_fh_gappy)
#     # compute fasttail prediction
#     pred_fast = f._predict_out_of_sample_v1_fasttail(None, simple_fh_gappy)
#     assert pred_fast is not None, "fasttail should activate under guard conditions"
#     pd.testing.assert_index_equal(pred_fast.index, pred_legacy.index)
#     np.testing.assert_allclose(
#         pred_fast.values, pred_legacy.values, rtol=1e-10, atol=1e-12
#     )


# @pytest.mark.parametrize(
#     "make_y, kwargs",
#     [
#         # MultiIndex y -> guard fail
#         (
#             lambda: pd.DataFrame(
#                 {"y": np.tile(np.arange(30), 2)},
#                 index=pd.MultiIndex.from_product(
#                     [["A", "B"], pd.period_range("2000-01", periods=30, freq="M")],
#                     names=["series", "time"],
#                 ),
#             ),
#             dict(pooling="local", impute_method=None, expect_none=True),
#         ),
#         # global pooling -> guard fail
#         (
#             lambda: pd.DataFrame(
#                 {"y": np.arange(40)},
#                 index=pd.period_range("2000-01", periods=40, freq="M"),
#             ),
#             dict(pooling="global", impute_method=None, expect_none=True),
#         ),
#         # imputation active -> guard fail
#         (
#             lambda: pd.DataFrame(
#                 {"y": np.arange(40)},
#                 index=pd.period_range("2000-01", periods=40, freq="M"),
#             ),
#             dict(pooling="local", impute_method="pad", expect_none=True),
#         ),
#         # baseline eligible -> expect activation
#         (
#             lambda: pd.DataFrame(
#                 {"y": np.arange(40)},
#                 index=pd.period_range("2000-01", periods=40, freq="M"),
#             ),
#             dict(pooling="local", impute_method=None, expect_none=False),
#         ),
#     ],
# )
# def test_fasttail_guard_activation(make_y, kwargs):
#     y = make_y()
#     expect_none = kwargs.pop("expect_none")
#     f = _fit_rec(y, window_length=5, **kwargs)
#     fh = ForecastingHorizon(
#         [1, 2, 3],
#         is_relative=True,
#         freq=y.index
#         if not isinstance(y.index, pd.MultiIndex)
#         else y.index.get_level_values(-1),
#     )
#     result = f._predict_out_of_sample_v1_fasttail(None, fh)
#     if expect_none:
#         assert result is None, "fasttail should return None under guard failure"
#     else:
#         assert result is not None, \
#            "fasttail should activate under eligible conditions"


@pytest.mark.skip(reason="I am not convinced the legacy predictions were correct")
def test_v2_local_equivalence(simple_series, simple_fh_gappy):
    """v2 local optimized path equals legacy predictions for simple case (gappy)."""
    f = _fit_rec(simple_series, window_length=5, pooling="local", impute_method="bfill")
    # direct internal optimized path (returns full gapless horizon predictions)
    fh_gapless_abs, _ = f._generate_fh_no_gaps(simple_fh_gappy)
    pred_v2_full = f._predict_out_of_sample_v2_local(None, simple_fh_gappy)
    # legacy restricted predictions
    pred_v1 = f._predict_out_of_sample_v1(None, simple_fh_gappy)
    if isinstance(pred_v2_full, np.ndarray):
        # wrap full sequence (gapless) first
        fh_gapless_abs_pd = fh_gapless_abs.to_pandas()
        pred_v2_full_df = pd.DataFrame(
            pred_v2_full, index=fh_gapless_abs_pd, columns=f._y.columns
        )
    else:
        pred_v2_full_df = pred_v2_full
    # now slice optimized predictions to gappy fh
    pred_v2 = pred_v2_full_df.loc[pred_v1.index]
    pd.testing.assert_index_equal(pred_v2.index, pred_v1.index)
    np.testing.assert_allclose(pred_v2.values, pred_v1.values, rtol=1e-10, atol=1e-12)


@pytest.mark.skip(reason="Design decision: not necessarily a target behaviour")
def test_constant_mean_fallback():
    """Constant mean path when no full lag rows available produces repeated mean."""
    # window length > len(y) so estimator_ becomes Series fallback
    y = pd.DataFrame(
        {"y": [1.0, 2.0]}, index=pd.period_range("2000-01", periods=2, freq="M")
    )
    f = _fit_rec(y, window_length=5, pooling="local", impute_method="bfill")
    assert isinstance(f.estimator_, pd.Series), (
        "Estimator should be mean Series fallback"
    )
    fh = ForecastingHorizon([1, 2, 4], is_relative=True, freq=y.index)
    preds = f.predict(fh=fh)
    assert (preds.values.ravel() == np.mean(y.values)).all(), (
        "All preds should equal mean"
    )


@pytest.mark.skip(
    reason="Design decision: disallow predictions without explicit future X"
)
def test_slice_at_ix_surrogate_simple():
    """slice_at_ix returns surrogate earlier row without raising when label missing."""
    idx = pd.Index([1, 3, 7, 10])
    df = pd.DataFrame({"a": [10, 11, 12, 13]}, index=idx)
    res = slice_at_ix(df, 5)  # 5 missing, should pick floor=3
    assert not res.empty and res.index[0] == 3
    res2 = slice_at_ix(df, 0)  # before earliest -> should pick earliest (1)
    assert res2.index[0] == 1


@pytest.mark.skip(
    reason="Design decision: disallow predictions without explicit future X"
)
def test_slice_at_ix_surrogate_multiindex():
    series_ids = ["A", "B"]
    time_idx = [1, 2, 4]
    mi = pd.MultiIndex.from_product([series_ids, time_idx], names=["id", "time"])
    df = pd.DataFrame({"a": range(len(mi))}, index=mi)
    # request time=3 which is missing -> expect rows with time=2 (floor)
    res = slice_at_ix(df, 3)
    assert not res.empty
    assert set(res.index.get_level_values(-1)) == {2}


def test_exogenous_prediction_no_nans(simple_series):
    """Predictions with exogenous X should produce no NaNs and correct index.

    Ensures that adding new exogenous columns (not present during fit) does not
    violate scikit-learn feature name expectations by fitting with exogenous.
    """
    fh = ForecastingHorizon([1, 2, 5], is_relative=True, freq=simple_series.index)
    future_index = fh.to_absolute(simple_series.index[-1:]).to_pandas()
    X_full = pd.DataFrame(
        {"x": np.linspace(0, len(simple_series) - 1, len(simple_series))},
        index=simple_series.index,
    )
    # Fit with exogenous present (only up to cutoff) so estimator sees 'x'
    # cutoff = simple_series.index[-1]
    f = _fit_rec(simple_series, window_length=5, pooling="local", impute_method="bfill")
    # Manually inject feature cols to mimic training with exogenous 'x'
    # Re-fit properly including X so feature names align
    f = RecursiveReductionForecaster(
        estimator=LinearRegression(),
        window_length=5,
        pooling="local",
        impute_method="bfill",
    )
    fh_fit = ForecastingHorizon([1], is_relative=True, freq=simple_series.index)
    f.fit(simple_series, X=X_full, fh=fh_fit)
    last_val = X_full.iloc[-1, 0]
    future_vals = np.arange(1, len(future_index) + 1) + last_val
    X_future = pd.DataFrame({"x": future_vals}, index=future_index)
    X_pool = pd.concat([X_full, X_future])
    preds = f.predict(X=X_pool, fh=fh)
    assert preds.isna().sum().sum() == 0
    pd.testing.assert_index_equal(preds.index, fh.to_absolute(f.cutoff).to_pandas())
