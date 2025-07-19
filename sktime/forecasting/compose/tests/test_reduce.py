#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test reduce."""

__author__ = ["Lovkush-A", "mloning", "LuisZugasti", "AyushmaanSeth"]

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    DirectReductionForecaster,
    DirectTabularRegressionForecaster,
    DirectTimeSeriesRegressionForecaster,
    DirRecTabularRegressionForecaster,
    DirRecTimeSeriesRegressionForecaster,
    MultioutputTabularRegressionForecaster,
    MultioutputTimeSeriesRegressionForecaster,
    RecursiveReductionForecaster,
    RecursiveTabularRegressionForecaster,
    RecursiveTimeSeriesRegressionForecaster,
    make_reduction,
)
from sktime.forecasting.compose._reduce import _sliding_window_transform
from sktime.forecasting.tests._config import TEST_OOS_FHS, TEST_WINDOW_LENGTHS_INT
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.regression.base import BaseRegressor
from sktime.regression.interval_based import TimeSeriesForestRegressor
from sktime.split import SlidingWindowSplitter, temporal_train_test_split
from sktime.split.tests.test_split import _get_windows
from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.panel.reduce import Tabularizer
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_fh

N_TIMEPOINTS = [13, 17]
N_VARIABLES = [1, 3]
STRATEGIES = ["recursive", "direct", "multioutput", "dirrec"]
FH = ForecastingHorizon(1)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"]),
    reason="run test only if forecasting or split module has changed",
)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS_INT)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("scitype", ["tabular-regressor", "time-series-regressor"])
def test_sliding_window_transform_against_cv(n_timepoints, window_length, fh, scitype):
    """Test sliding window transform against cv."""
    fh = check_fh(fh)
    y = pd.Series(_make_y(0, n_timepoints))
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length)
    xa, ya = _get_windows(cv, y)
    yb, xb = _sliding_window_transform(y, window_length, fh, scitype=scitype)
    np.testing.assert_array_equal(ya, yb)
    if scitype == "time-series-regressor":
        xb = xb.squeeze(axis=1)

    np.testing.assert_array_equal(xa, xb)


def _make_y_X(n_timepoints, n_variables):
    # We generate y and X values so that the y should always be greater
    # than its lagged values and the lagged and contemporaneous values of the
    # exogenous variables X
    assert n_variables < 10
    base = np.arange(n_timepoints)
    y = pd.Series(base + n_variables / 10)

    if n_variables > 1:
        X = np.column_stack([base + i / 10 for i in range(1, n_variables)])
        X = pd.DataFrame(X)
    else:
        X = None

    return y, X


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"]),
    reason="run test only if forecasting or split module has changed",
)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_variables", N_VARIABLES)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS_INT)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_sliding_window_transform_tabular(n_timepoints, window_length, n_variables, fh):
    """Test sliding window transform tabular."""
    y, X = _make_y_X(n_timepoints=n_timepoints, n_variables=n_variables)
    fh = check_fh(fh, enforce_relative=True)
    fh_max = fh[-1]
    effective_window_length = window_length + fh_max - 1

    yt, Xt = _sliding_window_transform(
        y, window_length=window_length, fh=fh, X=X, scitype="tabular-regressor"
    )
    assert yt.shape == (n_timepoints - effective_window_length, len(fh))

    # Check y values for first step in fh.
    actual = yt[:, 0]
    start = window_length + fh[0] - 1
    end = start + n_timepoints - window_length - fh_max + 1
    expected = y[np.arange(start, end)]
    np.testing.assert_array_equal(actual, expected)

    # The transformed Xt array contains lagged values for each variable, excluding the
    # n_variables + 1 contemporaneous values for the exogenous variables.
    # assert Xt.shape == (yt.shape[0], (window_length * n_variables) + n_variables - 1)
    assert Xt.shape == (yt.shape[0], window_length * n_variables)
    assert np.all(Xt < yt[:, [0]])


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"]),
    reason="run test only if forecasting or split module has changed",
)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_variables", N_VARIABLES)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS_INT)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_sliding_window_transform_panel(n_timepoints, window_length, n_variables, fh):
    """Test sliding window transform panel."""
    y, X = _make_y_X(n_timepoints=n_timepoints, n_variables=n_variables)
    fh = check_fh(fh, enforce_relative=True)
    fh_max = fh[-1]
    effective_window_length = window_length + fh_max - 1

    yt, Xt = _sliding_window_transform(
        y, window_length=window_length, X=X, fh=fh, scitype="time-series-regressor"
    )
    assert yt.shape == (n_timepoints - effective_window_length, len(fh))

    # Check y values.
    actual = yt[:, 0]
    start = window_length + fh[0] - 1
    end = start + n_timepoints - window_length - fh_max + 1
    expected = y[np.arange(start, end)]
    np.testing.assert_array_equal(actual, expected)

    # Given the input data, all of the value in the transformed Xt array should be
    # smaller than the transformed yt target array.
    assert Xt.shape == (yt.shape[0], n_variables, window_length)
    assert np.all(Xt < yt[:, np.newaxis, [0]])


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"]),
    reason="run test only if forecasting or split module has changed",
)
def test_sliding_window_transform_explicit():
    """Test sliding window transform explicit.

    testing with explicitly written down expected outputs intended to help future
    contributors understand the transformation
    """
    y = pd.Series(np.arange(9))
    X = pd.concat([y + 100, y + 200], axis=1)
    fh = ForecastingHorizon([1, 3], is_relative=True)
    window_length = 2

    yt_actual, Xt_tabular_actual = _sliding_window_transform(y, window_length, fh, X)
    _, Xt_time_series_actual = _sliding_window_transform(
        y, window_length, fh, X, scitype="time-series-regressor"
    )

    yt_expected = np.array([[2.0, 4.0], [3.0, 5.0], [4.0, 6.0], [5.0, 7.0], [6.0, 8.0]])

    Xt_tabular_expected = np.array(
        [
            [0.0, 1.0, 100.0, 101.0, 200.0, 201.0],
            [1.0, 2.0, 101.0, 102.0, 201.0, 202.0],
            [2.0, 3.0, 102.0, 103.0, 202.0, 203.0],
            [3.0, 4.0, 103.0, 104.0, 203.0, 204.0],
            [4.0, 5.0, 104.0, 105.0, 204.0, 205.0],
        ]
    )

    Xt_time_series_expected = np.array(
        [
            [[0.0, 1.0], [100.0, 101.0], [200.0, 201.0]],
            [[1.0, 2.0], [101.0, 102.0], [201.0, 202.0]],
            [[2.0, 3.0], [102.0, 103.0], [202.0, 203.0]],
            [[3.0, 4.0], [103.0, 104.0], [203.0, 204.0]],
            [[4.0, 5.0], [104.0, 105.0], [204.0, 205.0]],
        ]
    )

    np.testing.assert_array_equal(yt_actual, yt_expected)
    np.testing.assert_array_equal(Xt_tabular_actual, Xt_tabular_expected)
    np.testing.assert_array_equal(Xt_time_series_actual, Xt_time_series_expected)


def _make_y(start, end, method="linear-trend", slope=1):
    # generate test data
    if method == "linear-trend":
        y = np.arange(start, end) * slope
    else:
        raise ValueError("`method` not understood")
    return y


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"]),
    reason="run test only if forecasting or split module has changed",
)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS_INT)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "regressor, scitype",
    [
        (LinearRegression(), "tabular-regressor"),
        (make_pipeline(Tabularizer(), LinearRegression()), "time-series-regressor"),
    ],
)
@pytest.mark.parametrize(
    "method, slope",
    [
        ("linear-trend", 1),
        ("linear-trend", -3),
        ("linear-trend", 0),  # constant
    ],
)
def test_linear_extrapolation_endogenous_only(
    fh, window_length, strategy, method, slope, regressor, scitype
):
    """Test linear extrapolation endogenous only."""
    n_timepoints = 13
    y = _make_y(0, n_timepoints, method=method, slope=slope)
    y = pd.Series(y)
    fh = check_fh(fh)

    forecaster = make_reduction(
        regressor, scitype=scitype, window_length=window_length, strategy=strategy
    )
    forecaster.fit(y, fh=fh)
    actual = forecaster.predict()

    end = n_timepoints + max(fh) + 1
    expected = _make_y(n_timepoints, end, method=method, slope=slope)[fh.to_indexer()]
    np.testing.assert_almost_equal(actual, expected)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"]),
    reason="run test only if forecasting or split module has changed",
)
@pytest.mark.parametrize("fh", [1, 3, 5])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS_INT)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("scitype", ["time-series-regressor", "tabular-regressor"])
def test_dummy_regressor_mean_prediction_endogenous_only(
    fh, window_length, strategy, scitype
):
    """Test dummy regressor mean prediction endogenous_only.

    The DummyRegressor ignores the input feature data X, hence we can use it for testing
    reduction from forecasting to both tabular and time series regression. The
    DummyRegressor also supports the 'multioutput' strategy.
    """
    y = make_forecasting_problem()
    fh = check_fh(fh)
    y_train, y_test = temporal_train_test_split(y, fh=fh)

    regressor = DummyRegressor(strategy="mean")
    forecaster = make_reduction(
        regressor, scitype=scitype, window_length=window_length, strategy=strategy
    )
    forecaster.fit(y_train, fh=fh)
    actual = forecaster.predict()

    if strategy == "recursive":
        # For the recursive strategy, we always use the first-step ahead as the
        # target vector in the regression problem during training, regardless of the
        # actual forecasting horizon.
        effective_window_length = window_length
    else:
        # For the other strategies, we split the data taking into account the steps
        # ahead we want to predict.
        effective_window_length = window_length + max(fh) - 1

    # In the sliding-window transformation, the first values of the target series
    # make up the first window and are not used in the transformed target vector. So
    # the expected result should be the mean of the remaining values.
    expected = np.mean(y_train[effective_window_length:])
    np.testing.assert_array_almost_equal(actual, expected)


_REGISTRY = [
    ("tabular-regressor", "direct", DirectTabularRegressionForecaster),
    ("tabular-regressor", "recursive", RecursiveTabularRegressionForecaster),
    ("tabular-regressor", "multioutput", MultioutputTabularRegressionForecaster),
    ("tabular-regressor", "dirrec", DirRecTabularRegressionForecaster),
    ("time-series-regressor", "direct", DirectTimeSeriesRegressionForecaster),
    ("time-series-regressor", "recursive", RecursiveTimeSeriesRegressionForecaster),
    ("time-series-regressor", "multioutput", MultioutputTimeSeriesRegressionForecaster),
    ("time-series-regressor", "dirrec", DirRecTimeSeriesRegressionForecaster),
]


class _Recorder:
    # Helper class to record given data for later inspection.
    def fit(self, X, y):
        self.X_fit = X
        self.y_fit = y
        return self

    def predict(self, X):
        self.X_pred = X
        return np.ones(1)


class _TestTabularRegressor(BaseEstimator, RegressorMixin, _Recorder):
    pass


class _TestTimeSeriesRegressor(_Recorder, BaseRegressor):
    pass

    def _fit(self, X, y):
        """Empty method to satisfy abstract parent.

        Needs refactoring.
        """

    def _predict(self, X):
        """Empty method to satisfy abstract parent.

        Needs refactoring.
        """


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"]),
    reason="run test only if forecasting or split module has changed",
)
@pytest.mark.parametrize(
    "estimator", [_TestTabularRegressor(), _TestTimeSeriesRegressor()]
)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS_INT)
@pytest.mark.parametrize("strategy", ["recursive", "direct", "multioutput"])
def test_consistent_data_passing_to_component_estimators_in_fit_and_predict(
    estimator, window_length, strategy
):
    """Test consistent data passing to component estimators in fit and predict.

    We generate data that represents time points in its values, i.e. an array of values
    that increase in unit steps for each time point.
    """
    n_variables = 3
    n_timepoints = 10
    y, X = _make_y_X(n_timepoints, n_variables)
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, fh=FH)

    forecaster = make_reduction(
        estimator, strategy=strategy, window_length=window_length
    )
    forecaster.fit(y_train, X_train, fh=FH)
    forecaster.predict(X=X_test)

    # Get recorded data.
    if strategy == "direct":
        estimator_ = forecaster.estimators_[0]
    else:
        estimator_ = forecaster.estimator_

    X_fit = estimator_.X_fit
    y_fit = estimator_.y_fit
    X_pred = estimator_.X_pred

    # Format feature data into 3d array if the data is not in that format already.
    X_fit = X_fit.reshape(X_fit.shape[0], n_variables, -1)
    X_pred = X_pred.reshape(X_pred.shape[0], n_variables, -1)

    # Format target data into 2d array.
    y_fit = y_fit.reshape(y_fit.shape[0], -1)

    # Check that both fit and predict data have unit steps between them.
    assert np.allclose(np.diff(X_fit), 1)
    assert np.allclose(np.diff(X_pred), 1)

    # Check that predict data is a step ahead from last row in fit data.
    np.testing.assert_array_equal(X_pred, X_fit[[-1]] + 1)

    # Check that y values are further ahead than X values.
    assert np.all(X_fit < y_fit[:, np.newaxis, :])


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
@pytest.mark.parametrize("scitype, strategy, klass", _REGISTRY)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS_INT)
def test_make_reduction_construct_instance(scitype, strategy, klass, window_length):
    """Test make_reduction."""
    estimator = DummyRegressor()
    forecaster = make_reduction(
        estimator, window_length=window_length, scitype=scitype, strategy=strategy
    )
    assert isinstance(forecaster, klass)
    assert forecaster.get_params()["window_length"] == window_length


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.regression"]),
    reason="run test only if forecasting or regression module has changed",
)
@pytest.mark.parametrize(
    "estimator, scitype",
    [
        (LinearRegression(), "tabular-regressor"),
        (TimeSeriesForestRegressor(), "time-series-regressor"),
    ],
)
def test_make_reduction_infer_scitype(estimator, scitype):
    """Test make_reduction."""
    forecaster = make_reduction(estimator, scitype="infer")
    assert forecaster._estimator_scitype == scitype


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"]),
    reason="run test only if forecasting or split module has changed",
)
def test_make_reduction_infer_scitype_for_sklearn_pipeline():
    """Test make_reduction.

    The scitype of pipeline cannot be inferred here, as it may be used together with a
    tabular or time series regressor.
    """
    estimator = make_pipeline(Tabularizer(), LinearRegression())
    forecaster = make_reduction(estimator, scitype="infer")
    assert forecaster._estimator_scitype == "tabular-regressor"


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting", "sktime.split"])
    or not _check_soft_dependencies("catboost", severity="none"),
    reason="run test only if forecasting or split module has changed",
)
def test_make_reduction_with_catboost():
    """Test make_reduction with catboost.

    catboost is an example of a package that does not fully comply with the
    sklearn API. We therefore need to rely on the branch of scitype inference
    that assumes the estimator is a tabular regressor.
    """
    from catboost import CatBoostRegressor

    estimator = CatBoostRegressor(
        learning_rate=1, depth=6, loss_function="RMSE", verbose=False
    )

    forecaster = make_reduction(estimator, scitype="infer")
    assert forecaster._estimator_scitype == "tabular-regressor"

    fh = [1, 2, 3]
    y, X = make_forecasting_problem(make_X=True)
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, fh=fh)
    forecaster.fit(y_train, X_train, fh=fh).predict(fh, X_test)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_multioutput_direct_equivalence_tabular_linear_regression(fh):
    """Test multioutput and direct strategies with linear regression.

    Regressor should produce same predictions
    """
    y, X = make_forecasting_problem(make_X=True)
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, fh=fh)

    estimator = LinearRegression()
    direct = make_reduction(estimator, strategy="direct")
    multioutput = make_reduction(estimator, strategy="multioutput")

    y_pred_direct = direct.fit(y_train, X_train, fh=fh).predict(fh, X_test)
    y_pred_multioutput = multioutput.fit(y_train, X_train, fh=fh).predict(fh, X_test)

    np.testing.assert_array_almost_equal(
        y_pred_direct.to_numpy(), y_pred_multioutput.to_numpy()
    )


# this expected value created by Lovkush Agarwal by running code locally in Mar 2021
EXPECTED_AIRLINE_LINEAR_RECURSIVE = [
    397.28122475088117,
    391.0055770755232,
    382.85931770491493,
    376.75382498759643,
    421.3439733242519,
    483.7127665080476,
    506.5011555360703,
    485.95155173523494,
    414.41328025499604,
    371.2843322707713,
    379.5680077722808,
    406.146827316167,
    426.48249271837176,
    415.5337957767289,
    405.48715913377714,
    423.97150765765025,
    472.10998764155966,
    517.7763038626333,
    515.6077989417864,
    475.8615207069196,
    432.47049089698646,
    417.62468250043514,
    435.3174101071012,
    453.8693707695759,
]

# this expected value created by Lovkush Agarwal by running code locally in Mar 2021
EXPECTED_AIRLINE_LINEAR_DIRECT = [
    388.7894742436609,
    385.4311737990922,
    404.66760376792183,
    389.3921653574014,
    413.5415037170552,
    491.27471550855756,
    560.5985060880608,
    564.1354313250545,
    462.8049467298484,
    396.8247623180332,
    352.5416937680942,
    369.3915756974357,
    430.12889943026323,
    417.13419789042484,
    434.8091175980315,
    415.33997516059355,
    446.97711875155846,
    539.6761098618977,
    619.7204673400846,
    624.3153932803112,
    499.686252475341,
    422.0658526180952,
    373.3847171492921,
    388.8020135264563,
]


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
@pytest.mark.parametrize(
    "forecaster, expected",
    [
        (
            DirectTabularRegressionForecaster(LinearRegression()),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        # multioutput should behave the same as direct with linear regression estimator
        # hence the reason for the same expected predictions
        (
            MultioutputTabularRegressionForecaster(LinearRegression()),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        (
            RecursiveTabularRegressionForecaster(LinearRegression()),
            EXPECTED_AIRLINE_LINEAR_RECURSIVE,
        ),
        (
            RecursiveTabularRegressionForecaster(LinearRegression(), pooling="global"),
            EXPECTED_AIRLINE_LINEAR_RECURSIVE,
        ),
        (
            DirectTimeSeriesRegressionForecaster(
                make_pipeline(Tabularizer(), LinearRegression())
            ),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        # multioutput should behave the same as direct with linear regression estimator
        # hence the reason for the same expected predictions
        (
            MultioutputTimeSeriesRegressionForecaster(
                make_pipeline(Tabularizer(), LinearRegression())
            ),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        (
            RecursiveTimeSeriesRegressionForecaster(
                make_pipeline(Tabularizer(), LinearRegression())
            ),
            EXPECTED_AIRLINE_LINEAR_RECURSIVE,
        ),
    ],
)
def test_reductions_airline_data(forecaster, expected):
    """Test reduction forecasters.

    Test reduction forecasters by making prediction on airline dataset using linear
    estimators. Predictions compared with values calculated by Lovkush Agarwal on their
    local machine in Mar 2021
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    actual = forecaster.fit(y_train, fh=fh).predict(fh)

    np.testing.assert_almost_equal(actual, expected)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
def test_dirrec_against_recursive_accumulated_error():
    """Test recursive and dirrec regressor strategies.

    dirrec regressor should produce lower error due to less cumulative error
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    estimator = LinearRegression()
    recursive = make_reduction(
        estimator, scitype="tabular-regressor", strategy="recursive"
    )
    dirrec = make_reduction(estimator, scitype="tabular-regressor", strategy="dirrec")

    preds_recursive = recursive.fit(y_train, fh=fh).predict(fh)
    preds_dirrec = dirrec.fit(y_train, fh=fh).predict(fh)

    assert mean_absolute_percentage_error(
        y_test, preds_dirrec
    ) < mean_absolute_percentage_error(y_test, preds_recursive)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
def test_direct_vs_recursive():
    """Test reduction forecasters.

    Test reduction forecasters by making prediction on airline dataset using linear
    estimators. Wenn windows_identical = False, all observations should be considered
    (see documentation in make_reduction function), so results for direct and recursive
    forecasting should match for the first forecasting horizon. With the
    windows_identical
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster_dir_max = DirectTabularRegressionForecaster(
        LinearRegression(), windows_identical=False
    )
    forecaster_dir_spec = DirectTabularRegressionForecaster(
        LinearRegression(), windows_identical=True
    )
    forecaster_rec_max = RecursiveTabularRegressionForecaster(LinearRegression())
    forecaster_rec_spec = RecursiveTabularRegressionForecaster(LinearRegression())

    pred_dir_max = forecaster_dir_max.fit(y_train, fh=fh).predict(fh)
    pred_dir_spec = forecaster_dir_spec.fit(y_train, fh=fh).predict(fh)
    pred_rec_max = forecaster_rec_max.fit(y_train, fh=fh).predict(fh)
    pred_rec_spec = forecaster_rec_spec.fit(y_train, fh=fh).predict(fh)

    assert pred_dir_max.head(1).equals(pred_rec_max.head(1))
    assert pred_dir_max.head(1).equals(pred_rec_spec.head(1))
    assert not pred_dir_max.head(1).equals(pred_dir_spec.head(1))


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
def test_recursive_reducer_X_not_fit_to_fh():
    """Test recursive reducer with X that do not fit the fh.

    I.e., either X is longer or smaller than max_fh
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)
    X_train = y_train
    X_test = y_test

    forecaster = make_reduction(
        LinearRegression(), window_length=2, strategy="recursive"
    )
    forecaster.fit(y_train, X_train)

    pred1 = forecaster.predict(X=X_test[:1], fh=[1, 2, 3])
    assert pred1.shape == (3,)
    pred2 = forecaster.predict(X=X_test[:2], fh=[1, 2, 3])
    assert pred2.shape == (3,)
    pred3 = forecaster.predict(X=X_test[:3], fh=[1, 2, 3])
    assert pred3.shape == (3,)
    pred4 = forecaster.predict(X=X_test, fh=[1])
    assert pred4.shape == (1,)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"])
    or not _check_soft_dependencies("skpro", severity="none"),
    reason="run test only if reduce module has changed",
)
def test_make_reduction_proba():
    """Test direct reduction via make_reduction with skpro proba regressors."""
    from skpro.regression.dummy import DummyProbaRegressor

    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = make_reduction(DummyProbaRegressor(), strategy="direct")
    y_pred = forecaster.fit(y_train, fh=fh).predict(fh)

    assert y_pred.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
@pytest.mark.parametrize("strategy", [("direct"), ("recursive")])
def test_reduction_without_X(strategy):
    """Test DirectReductionForecaster with manual calculation (no exogenous)."""

    y = np.array([1, 2, 3, 4]).reshape(-1, 1)

    X_manual = np.array([[1, 2], [2, 3]])  # t-2, t-1 steps
    y_manual = np.array([3, 4])  # step t

    manual_reg = LinearRegression().fit(X_manual, y_manual)

    forecaster = make_reduction(
        LinearRegression(),
        window_length=2,
        strategy=strategy,
        scitype="tabular-regressor",
        pooling="local",
    )
    forecaster.fit(y, fh=[1])

    manual_pred = manual_reg.predict([[3, 4]])
    forecaster_pred = forecaster.predict()
    assert np.allclose(forecaster_pred, manual_pred)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
@pytest.mark.parametrize(
    "x_treatment",
    ["concurrent", "shifted"],
)
def test_direct_reduction_with_X(x_treatment):
    """Test DirectReductionForecaster with exogenous variables."""
    X = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
    y = np.array([1, 2, 3, 4]).reshape(-1, 1)

    if x_treatment == "concurrent":
        X_manual = np.hstack(
            [
                y[:2],
                y[1:3],
                X[2:4],  # X(t+h) for concurrent.
            ]
        )
    else:
        X_manual = np.hstack(
            [
                y[:2],
                y[1:3],
                X[1:3],  # Use X(t) for shifted.
            ]
        )

    y_manual = np.array([3, 4])
    lr = LinearRegression()

    forecaster = DirectReductionForecaster(
        LinearRegression(), window_length=2, X_treatment=x_treatment
    )
    forecaster.fit(y, X=X, fh=[1])
    lr.fit(X_manual, y_manual)

    input_X = X[3:4]
    input_y = y[2:4].reshape(1, -1)
    y_pred = forecaster.predict(X=X)
    lr_pred = lr.predict([np.hstack([input_y, input_X]).flatten()])

    assert np.isclose(y_pred, lr_pred, rtol=1e-3)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.compose._reduce"]),
    reason="run test only if reduce module has changed",
)
def test_recursive_reduction_with_X():
    """Test RecursiveReductionForecaster with exogenous variables"""

    y = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3])
    X = pd.DataFrame(
        {"x1": [10, 30, 50, 70], "x2": [20, 40, 60, 80]}, index=[0, 1, 2, 3]
    )
    window_length = 2

    # Construct rolling window manually
    y_rolled = np.column_stack(
        [y.shift(i).values[window_length:] for i in range(window_length, 0, -1)]
    )
    X_manual = np.hstack([y_rolled, X.iloc[window_length:].values])
    y_manual = y.iloc[window_length:].values

    manual_lr = LinearRegression().fit(X_manual, y_manual)

    forecaster = RecursiveReductionForecaster(
        estimator=LinearRegression(), window_length=window_length
    )
    forecaster.fit(y, X=X, fh=ForecastingHorizon([1], is_relative=True))

    # Future Exogenous Data
    X_new = pd.DataFrame([[90, 100]], index=[y.index[-1] + 1], columns=["x1", "x2"])

    y_pred = forecaster.predict(X=X_new)
    last_window = y.iloc[-window_length:].values.reshape(1, -1)
    manual_input = np.hstack([last_window, X_new.values])
    manual_pred = manual_lr.predict(manual_input)

    assert np.allclose(y_pred, manual_pred)
