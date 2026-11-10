# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for native STR, including fixed reference results from R stR."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal

from sktime.transformations.detrend import STR

# Fixed reference tables from R 4.3.3 / Matrix 1.6-5, generated independently
# of the Python implementation. The R code below can regenerate these tables
# using robjhyndman/stR at c6bb49e7c6e42ea2d965195b9546bcb0edcae842.
# Run it in R with Matrix installed; R is not required to run these tests.
# library(compiler)
# library(Matrix)
# base <- paste0("https://raw.githubusercontent.com/robjhyndman/stR/",
#                "c6bb49e7c6e42ea2d965195b9546bcb0edcae842/R/")
# for (file in c("STR.R", "lmSolver.R")) {
#   local <- tempfile(fileext=".R")
#   download.file(paste0(base, file), local, quiet=TRUE)
#   source(local)
#   unlink(local)
# }
# .solve.dgC.chol <- Matrix:::.solve.dgC.chol
# make_predictor <- function(n, p, nk, nt, weights, trend=FALSE) {
#   list(name=if (trend) "trend" else paste0("seasonal_", p),
#        data=rep(1, n), times=0:(n-1),
#        seasons=if (trend) rep(1, n) else (1:n) %% p,
#        timeKnots=seq(0, n-1, length.out=nt),
#        seasonalStructure=list(segments=list(c(0, p)),
#          sKnots=if (trend) list(c(1,0)) else
#            c(as.list(seq(p/nk, p-p/nk, length.out=nk-1)), list(c(p,0)))),
#        lambdas=weights)
# }
# for (case in 1:2) {
#   n <- 24
#   t <- 0:(n-1)
#   y <- 3 + .07*t + (1+.02*t)*sin(2*pi*(t+1)/3) + .15*cos(t*1.7)
#   periods <- if (case==1) c(3) else c(3,6)
#   if (case==2) y <- y + .4*cos(2*pi*(t+1)/6)
#   predictors <- list(make_predictor(n,1,1,6,c(2,0,0),TRUE))
#   for (p in periods) {
#     predictors[[length(predictors)+1]] <- make_predictor(n,p,3,4,c(1.5,.3,.8))
#   }
#   result <- STRmodel(y, predictors)
#   output <- data.frame(data=y,
#     trend=as.vector(result$output$predictors[[1]]$data),
#     resid=as.vector(result$output$random$data))
#   for (i in seq_along(periods)) {
#     output[[paste0("seasonal_",periods[i])]] <-
#       as.vector(result$output$predictors[[i+1]]$data)
#   }
#   print(output, digits=16)
# }
_R_REFERENCE = {
    1: pd.DataFrame(
        [
            [4.01602540378444, 3.16627639546509, 0.516681385478735, 0.333067622840614],
            [2.16732741399554, 3.183648783103, -0.726195545959205, -0.290125823148255],
            [2.99498027111308, 3.20102117074092, -0.17892291696722, -0.027117982660616],
            [4.18468358941845, 3.21839355837883, 0.660023461523383, 0.306266569516239],
            [
                2.47510218746528,
                3.23576594601674,
                -0.476668590956729,
                -0.283995167594734,
            ],
            [
                3.25969821459728,
                3.28408124910357,
                -0.0179354496165369,
                -0.00644758488976116,
            ],
            [4.28280860443449, 3.37881092536378, 0.624532162878848, 0.279465516191864],
            [2.6206415841069, 3.47354060162399, -0.575034505475875, -0.277864512041214],
            [
                3.63675559886797,
                3.56827027788419,
                0.0573699636479135,
                0.0111153573358657,
            ],
            [4.51422035940349, 3.6629999541444, 0.571184350143882, 0.280036055115212],
            [
                2.61949501475094,
                3.72559219318953,
                -0.808215224333977,
                -0.297881954104614,
            ],
            [
                3.91832560140412,
                3.78015007293089,
                0.134356873273567,
                0.00381865519966826,
            ],
            [4.91692412719246, 3.83470795267224, 0.767395090171997, 0.314821084348215],
            [2.66969576159222, 3.8892658324136, -0.894199789619964, -0.32537028120142],
            [
                4.01537195314257,
                3.95053018416224,
                0.0683198159168614,
                -0.0034780469365292,
            ],
            [5.31583029172936, 4.03862042393999, 0.927603754208155, 0.349606113581218],
            [
                2.90539875977229,
                4.12671066371774,
                -0.875748160869978,
                -0.345563743075471,
            ],
            [
                4.06841402060424,
                4.21480090349548,
                -0.144330810514035,
                -0.00205607237720888,
            ],
            [5.54057362443644, 4.30289114327323, 0.896420681952831, 0.341261799210374],
            [3.22998588621353, 4.35540215096578, -0.78519108808237, -0.34022517666988],
            [4.27271445878231, 4.38419367060154, -0.11782005535776, 0.0063408435385256],
            [5.6374552123165, 4.4129851902373, 0.896943705190149, 0.327526316889049],
            [
                3.43626320564234,
                4.44177670987306,
                -0.670626893966427,
                -0.334886610264289,
            ],
            [4.63536377629683, 4.47056822950881, 0.150057787333759, 0.0147377594542601],
        ],
        columns=["data", "trend", "resid", "seasonal_3"],
    ),
    2: pd.DataFrame(
        [
            [
                4.21602540378444,
                3.33452636087465,
                0.563731736092759,
                0.322426111056143,
                -0.00465880423911308,
            ],
            [
                1.96732741399554,
                3.30569430800906,
                -0.537098241200605,
                -0.24426145121127,
                -0.55700720160164,
            ],
            [
                2.59498027111308,
                3.27686225514347,
                -0.356836861350046,
                -0.0626239860168023,
                -0.262421136663541,
            ],
            [
                3.98468358941845,
                3.24803020227788,
                0.439674479969312,
                0.299780197231272,
                -0.00280129006001133,
            ],
            [
                2.67510218746528,
                3.21919814941229,
                -0.530618039469352,
                -0.24559164504574,
                0.232113722568084,
            ],
            [
                3.65969821459728,
                3.24416071487999,
                0.0100039626748996,
                -0.0386478783574614,
                0.444181415399846,
            ],
            [
                4.48280860443449,
                3.34981520784764,
                0.848399018881329,
                0.277134283406401,
                0.00746009429912441,
            ],
            [
                2.4206415841069,
                3.45546970081528,
                -0.41645858915748,
                -0.24692183888021,
                -0.371447688670694,
            ],
            [
                3.23675559886797,
                3.56112419378293,
                -0.123210858725902,
                -0.0176764648691817,
                -0.183481271319872,
            ],
            [
                4.31422035940349,
                3.66677868675057,
                0.364047350290247,
                0.277802683244137,
                0.00559163911853952,
            ],
            [
                2.81949501475094,
                3.72211838905855,
                -0.814536951541525,
                -0.268019222426811,
                0.179932799660726,
            ],
            [
                4.31832560140412,
                3.7648793937016,
                0.256909261235199,
                -0.0207426047493932,
                0.317279551216712,
            ],
            [
                5.11692412719246,
                3.80764039834466,
                1.03511569101472,
                0.307613975160129,
                -0.0334459373270551,
            ],
            [
                2.46969576159222,
                3.85040140298772,
                -0.714458423498916,
                -0.294764374462593,
                -0.371482843433991,
            ],
            [
                3.61537195314257,
                3.90801425227979,
                -0.135035274610773,
                -0.0238087446296047,
                -0.13379827989684,
            ],
            [
                5.11583029172936,
                4.02503448016791,
                0.625178434295648,
                0.337425267076122,
                0.128192110189681,
            ],
            [
                3.10539875977229,
                4.14205470805603,
                -0.918115368497466,
                -0.314625815540639,
                0.196085235754366,
            ],
            [
                4.46841402060424,
                4.25907493594415,
                -0.0519256393255656,
                -0.0147486506228152,
                0.276013374608474,
            ],
            [
                5.74057362443644,
                4.37609516383226,
                1.11058581545897,
                0.320299740941974,
                -0.0664070957967699,
            ],
            [
                3.02998588621353,
                4.40861696682436,
                -0.623632368893034,
                -0.310394268266614,
                -0.444604443451182,
            ],
            [
                3.87271445878231,
                4.38480648655245,
                -0.350649441056955,
                0.00401243049357508,
                -0.165455017206765,
            ],
            [
                5.4374552123165,
                4.36099600628054,
                0.648800589753467,
                0.297307112551558,
                0.130351503730935,
            ],
            [
                3.63626320564234,
                4.33718552600862,
                -0.643278807070514,
                -0.306162720992588,
                0.248519207696816,
            ],
            [
                5.03536377629683,
                4.31337504573671,
                0.313408524731573,
                0.0227735116099654,
                0.385806694218584,
            ],
        ],
        columns=["data", "trend", "resid", "seasonal_3", "seasonal_6"],
    ),
}


@pytest.mark.parametrize("case,periods", [(1, [3]), (2, [3, 6])])
def test_r_reference(case, periods):
    """Match STRmodel from R for one and multiple time-varying seasonalities."""
    reference = _R_REFERENCE[case]
    transformer = STR(
        periods=periods,
        lambdas=[2] + [1.5, 0.3, 0.8] * len(periods),
        trend_knots=6,
        seasonal_time_knots=4,
        seasonal_knots=3,
        return_components=True,
    )
    result = transformer.fit_transform(reference["data"])
    assert_allclose(result, reference.drop(columns="data"), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("components", [False, True])
@pytest.mark.parametrize("index_type", ["range", "period", "datetime"])
def test_reconstruction(components, index_type):
    """Preserve input labels and reconstruct the observed series exactly."""
    index = {
        "range": pd.RangeIndex(48),
        "period": pd.period_range("2000-01", periods=48, freq="M"),
        "datetime": pd.date_range("2000-01-01", periods=48, freq="D"),
    }[index_type]
    t = np.arange(48)
    X = pd.Series(0.1 * t + np.sin(2 * np.pi * t / 6), index=index, name="signal")
    transformer = STR(periods=[2, 6], return_components=components)
    result = transformer.fit_transform(X)
    assert result.index.equals(X.index)
    assert_allclose(
        transformer.trend_ + transformer.seasonal_.sum(axis=1) + transformer.resid_, X
    )
    restored = transformer.inverse_transform(result)
    if components:
        assert list(restored.columns) == [X.name]
        restored = restored.iloc[:, 0]
    assert_series_equal(restored, X, atol=1e-10, rtol=1e-10)
    if components:
        assert list(result.columns) == ["trend", "resid", "seasonal_2", "seasonal_6"]


def test_linear_trend_and_varying_seasonality():
    """Recover a linear trend and seasonal amplitude that grows linearly."""
    t = np.arange(90)
    trend = 4 + 0.03 * t
    seasonal = (1 + 0.02 * t) * np.sin(2 * np.pi * (t + 1) / 3)
    transformer = STR(periods=3, lambdas=[1, 1, 0.0001, 0.0001])
    transformer.fit(pd.Series(trend + seasonal))
    assert_allclose(transformer.trend_, trend, atol=1e-6)
    assert_allclose(transformer.seasonal_["seasonal_3"], seasonal, atol=1e-6)


def test_transform_new_values_does_not_mutate():
    """Recompute for changed values and return requested indices only."""
    X = pd.Series(np.sin(np.arange(36)), name="signal")
    transformer = STR(periods=3, return_components=True).fit(X)
    original = transformer.transform(X)
    subset = X.iloc[12:24] + 2
    result = transformer.transform(subset)
    assert result.index.equals(subset.index)
    assert_allclose(result.sum(axis=1), subset)
    assert_frame_equal(transformer.transform(X), original)
    changed = X + 1
    assert_allclose(transformer.transform(changed).sum(axis=1), changed)


@pytest.mark.parametrize("components", [False, True])
def test_forecasting_pipeline(components):
    """Both output modes work in a forecasting pipeline."""
    from sktime.forecasting.naive import NaiveForecaster

    X = pd.Series(
        np.sin(np.arange(48)), index=pd.period_range("2000", periods=48, freq="M")
    )
    transformer = STR(periods=[2, 6], return_components=components)
    forecaster = transformer * NaiveForecaster()
    forecaster.fit(X, fh=[1, 2, 7])
    prediction = forecaster.predict()
    assert prediction.index.equals(pd.PeriodIndex([X.index[-1] + h for h in [1, 2, 7]]))
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()


def test_inverse_future_repeats_last_cycle():
    """Future inverse adds each fitted season's last cycle at the right phase."""
    X = pd.Series(np.sin(np.arange(48)))
    transformer = STR(periods=[2, 6]).fit(X)
    horizon = pd.Index([48, 49, 54])
    result = transformer.inverse_transform(pd.Series(0.0, index=horizon))
    expected = sum(
        transformer.seasonal_[f"seasonal_{p}"]
        .iloc[-p:]
        .to_numpy()[np.array([0, 1, 6]) % p]
        for p in [2, 6]
    )
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "params,match",
    [
        ({"periods": []}, "periods"),
        ({"periods": [2, 2]}, "periods"),
        ({"periods": 1}, "periods"),
        ({"periods": 2.5}, "periods"),
        ({"periods": True}, "periods"),
        ({"periods": [3, 30]}, "two full cycles"),
        ({"trend_knots": 2}, "trend_knots"),
        ({"seasonal_time_knots": 2}, "seasonal_time_knots"),
        ({"seasonal_knots": [3, 4]}, "seasonal_knots"),
        ({"seasonal_knots": 1}, "seasonal_knots"),
        ({"lambdas": 0}, "lambdas"),
        ({"lambdas": np.inf}, "lambdas"),
        ({"lambdas": [1, 2]}, "lambdas"),
    ],
)
def test_invalid_parameters(params, match):
    """Reject invalid parameters with actionable messages."""
    with pytest.raises(ValueError, match=match):
        STR(**params).fit(pd.Series(np.arange(48, dtype=float)))


def test_nonfinite_input():
    """Reject infinities before passing them to the linear solver."""
    X = pd.Series(np.arange(48, dtype=float))
    X.iloc[5] = np.inf
    with pytest.raises(ValueError, match="finite"):
        STR(periods=3).fit(X)
