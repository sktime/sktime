#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Tomasz Chodakowski", "Ryan Kuhns"]

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sktime.utils._testing.series import _make_series
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteScaledError,
    MedianAbsoluteScaledError,
    MeanSquaredScaledError,
    MedianSquaredScaledError,
    MeanAbsoluteError,
    MeanSquaredError,
    MedianAbsoluteError,
    MedianSquaredError,
    MeanAbsolutePercentageError,
    MedianAbsolutePercentageError,
    MeanSquaredPercentageError,
    MedianSquaredPercentageError,
    MeanRelativeAbsoluteError,
    MedianRelativeAbsoluteError,
    GeometricMeanRelativeAbsoluteError,
    GeometricMeanRelativeSquaredError,
    MeanAsymmetricError,
    RelativeLoss,
    mean_absolute_scaled_error,
    median_absolute_scaled_error,
    mean_squared_scaled_error,
    median_squared_scaled_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    median_squared_error,
    mean_absolute_percentage_error,
    median_absolute_percentage_error,
    mean_squared_percentage_error,
    median_squared_percentage_error,
    mean_relative_absolute_error,
    median_relative_absolute_error,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
    mean_asymmetric_error,
    relative_loss,
)
from sktime.performance_metrics.tests._config import RANDOM_SEED

# For multiple comparisons of equality between functions and classes
rng = np.random.default_rng(RANDOM_SEED)
RANDOM_STATES = rng.integers(0, 1000000, size=5).tolist()

# Create specific test series to verify calculated performance metrics match
# those calculated externally
Y1 = np.array(
    [
        0.626832772836215,
        0.783382993377663,
        0.745780385700732,
        1.06737808331213,
        1.69664933579028,
        2.08627141338732,
        1.78023192557434,
        1.58568920200064,
        2.08902410668301,
        2.51472070324453,
        2.47425419784015,
        2.27275916300358,
        1.92803852608368,
        1.64662766528414,
        1.7028471682496,
        1.62051042240568,
        2.03642032341352,
        2.36019377457168,
        2.39730479510699,
        2.69699728045652,
        2.41172828049954,
        2.37679353181132,
        1.99603448413176,
        2.53946033171028,
        2.16285521091308,
        1.70889477546947,
        1.52488156869114,
        1.8369477471545,
        1.8225935878131,
        1.64685504990138,
        1.36106553603259,
        1.20252674753628,
        1.33235953453508,
        1.70560866839458,
        2.25722026784685,
        1.84446872239422,
    ]
)

Y2 = pd.Series(
    [
        0.982136629140069,
        1.45950325745833,
        1.42708285946536,
        2.10474124388042,
        2.12958738712948,
        1.94254184770726,
        2.24111458763484,
        2.68784805815518,
        2.97248086366361,
        3.27426914233203,
        3.16674535150384,
        2.933698752984,
        3.18393847027259,
        3.43030921792323,
        3.21901076902567,
        2.51266154720592,
        2.52702260323378,
        2.4241798970835,
        1.91495784087606,
        1.49993972682056,
        1.66460722130508,
        1.72380847201769,
        1.45265679700175,
        1.54961689438936,
        1.40262473301413,
        1.50833698230433,
        1.17807171492728,
        1.37642259034361,
        1.19122274092639,
        1.72766650406602,
        2.01019283258555,
        1.70144149287405,
        1.40552850108184,
        1.22336047820607,
        1.58882703694742,
        1.68674857175401,
    ]
)
# Data for this test case borrower from Rob Hyndman's excel workbook
# demonstrating how to calculate MASE
Y3 = np.array(
    [
        0,
        2,
        0,
        1,
        0,
        11,
        0,
        0,
        0,
        0,
        2,
        0,
        6,
        3,
        0,
        0,
        0,
        0,
        0,
        7,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
    ]
)
Y1_TRAIN, Y1_TEST = Y1[:24], Y1[24:]
Y2_TRAIN, Y2_TEST = Y2[:24], Y2[24:]
Y3_TRAIN, Y3_TEST = Y3[:24], Y3[24:]

Y_TEST_CASES = {
    "test_case_1": {"train": Y1_TRAIN, "test": Y1_TEST},
    "test_case_2": {"train": Y2_TRAIN, "test": Y2_TEST},
    "test_case_3": {"train": Y3_TRAIN, "test": Y3_TEST},
    # Multivariate test case
    "test_case_4": {
        "train": np.vstack([Y1_TRAIN, Y2_TRAIN]),
        "test": np.vstack([Y1_TEST, Y2_TEST]),
    },
}

# Dictionary mapping functions to the true loss values to verify the sktime
# metrics are performing as expected. True loss values were calculated
# manually outside of sktime in Excel.
LOSS_RESULTS = {
    "mean_absolute_scaled_error": {
        "test_case_1": 1.044427857,
        "test_case_2": 0.950832524,
        "test_case_3": 0.33045977,
        "func": mean_absolute_scaled_error,
        "class": MeanAbsoluteScaledError(),
    },
    "median_absolute_scaled_error": {
        "test_case_1": 0.997448587,
        "test_case_2": 0.975921875,
        "test_case_3": 1.0,
        "func": median_absolute_scaled_error,
        "class": MedianAbsoluteScaledError(),
    },
    "root_mean_squared_scaled_error": {
        "test_case_1": 1.001351033,
        "test_case_2": 0.854561506,
        "test_case_3": 0.289374954,
        "func": mean_squared_scaled_error,
        "class": MeanSquaredScaledError(square_root=True),
    },
    "root_median_squared_scaled_error": {
        "test_case_1": 0.998411526,
        "test_case_2": 0.990760662,
        "test_case_3": 1.0,
        "func": median_squared_scaled_error,
        "class": MedianSquaredScaledError(square_root=True),
    },
    "mean_absolute_error": {
        "test_case_1": 0.285709251,
        "test_case_2": 0.252975912,
        "test_case_3": 0.833333333,
        "func": mean_absolute_error,
        "class": MeanAbsoluteError(),
    },
    "mean_squared_error": {
        "test_case_1": 0.103989049,
        "test_case_2": 0.07852696,
        "test_case_3": 1.5,
        "func": mean_squared_error,
        "class": MeanSquaredError(),
    },
    "root_mean_squared_error": {
        "test_case_1": 0.322473331,
        "test_case_2": 0.280226623,
        "test_case_3": 1.224744871,
        "func": mean_squared_error,
        "class": MeanSquaredError(square_root=True),
    },
    "median_absolute_error": {
        "test_case_1": 0.298927846,
        "test_case_2": 0.240438602,
        "test_case_3": 1.0,
        "func": median_absolute_error,
        "class": MedianAbsoluteError(),
    },
    "median_squared_error": {
        "test_case_1": 0.089530473,
        "test_case_2": 0.059582098,
        "test_case_3": 1.0,
        "func": median_squared_error,
        "class": MedianSquaredError(),
    },
    "root_median_squared_error": {
        "test_case_1": 0.299216432,
        "test_case_2": 0.244094445,
        "test_case_3": 1.0,
        "func": median_squared_error,
        "class": MedianSquaredError(square_root=True),
    },
    "symmetric_mean_absolute_percentage_error": {
        "test_case_1": 0.16206745335345693,
        "test_case_2": 0.17096048184064724,
        "test_case_3": 1.0833333333333333,
        "func": mean_absolute_percentage_error,
        "class": MeanAbsolutePercentageError(symmetric=True),
    },
    "symmetric_median_absolute_percentage_error": {
        "test_case_1": 0.17291559217102262,
        "test_case_2": 0.15323286657516913,
        "test_case_3": 1.5,
        "func": median_absolute_percentage_error,
        "class": MedianAbsolutePercentageError(symmetric=True),
    },
    "mean_absolute_percentage_error": {
        "test_case_1": 0.16426360194846226,
        "test_case_2": 0.16956968442429066,
        "test_case_3": 1125899906842624.2,
        "func": mean_absolute_percentage_error,
        "class": MeanAbsolutePercentageError(symmetric=False),
    },
    "median_absolute_percentage_error": {
        "test_case_1": 0.17200352348889714,
        "test_case_2": 0.1521891319356885,
        "test_case_3": 1.0,
        "func": median_absolute_percentage_error,
        "class": MedianAbsolutePercentageError(symmetric=False),
    },
    "mean_squared_percentage_error": {
        "test_case_1": 0.03203423036447087,
        "test_case_2": 0.03427486821803671,
        "test_case_3": 5.070602400912918e30,
        "func": mean_squared_percentage_error,
        "class": MeanSquaredPercentageError(symmetric=False),
    },
    "median_squared_percentage_error": {
        "test_case_1": 0.029589708748632582,
        "test_case_2": 0.023172298452886965,
        "test_case_3": 1.0,
        "func": median_squared_percentage_error,
        "class": MedianSquaredPercentageError(symmetric=False),
    },
    "root_mean_squared_percentage_error": {
        "test_case_1": 0.17898108940463758,
        "test_case_2": 0.18513472990780716,
        "test_case_3": 2251799813685248.0,
        "func": mean_squared_percentage_error,
        "class": MeanSquaredPercentageError(square_root=True, symmetric=False),
    },
    "root_median_squared_percentage_error": {
        "test_case_1": 0.17201659439900727,
        "test_case_2": 0.15222450017289255,
        "test_case_3": 1.0,
        "func": median_squared_percentage_error,
        "class": MedianSquaredPercentageError(square_root=True, symmetric=False),
    },
    "mean_relative_absolute_error": {
        "test_case_1": 0.485695805,
        "test_case_2": 0.477896036,
        "test_case_3": 0.875,
        "func": mean_relative_absolute_error,
        "class": MeanRelativeAbsoluteError(),
    },
    "median_relative_absolute_error": {
        "test_case_1": 0.411364556,
        "test_case_2": 0.453437859,
        "test_case_3": 1.0,
        "func": median_relative_absolute_error,
        "class": MedianRelativeAbsoluteError(),
    },
    "geometric_mean_relative_absolute_error": {
        "test_case_1": 0.363521894,
        "test_case_2": 0.402438951,
        "test_case_3": 3.6839e-07,
        "func": geometric_mean_relative_absolute_error,
        "class": GeometricMeanRelativeAbsoluteError(),
    },
    "geometric_mean_relative_squared_error": {
        "test_case_1": 0.132148167,
        "test_case_2": 0.161957109,
        "test_case_3": 4.517843023201426e-07,
        "func": geometric_mean_relative_squared_error,
        "class": GeometricMeanRelativeSquaredError(),
    },
    "mean_aymmetric_error": {
        "test_case_1": 0.17139968,
        "test_case_2": 0.163956601,
        "test_case_3": 1.000000,
        "func": mean_asymmetric_error,
        "class": MeanAsymmetricError(),
    },
    "relative_loss": {
        "test_case_1": 0.442644622,
        "test_case_2": 0.416852592,
        "test_case_3": 1.315789474,
        "func": relative_loss,
        "class": RelativeLoss(),
    },
}


def _call_metrics(metric_func, metric_class, y_true, y_pred, y_train, y_pred_benchmark):
    """Call function and class metrics and return results"""
    class_attrs = metric_class.get_params()
    function_metric = metric_func(
        y_true,
        y_pred,
        y_train=y_train,
        y_pred_benchmark=y_pred_benchmark,
        **class_attrs,
    )
    class_metric = metric_class(
        y_true,
        y_pred,
        y_train=y_train,
        y_pred_benchmark=y_pred_benchmark,
    )
    return function_metric, class_metric


@pytest.mark.parametrize("metric_func_name", LOSS_RESULTS.keys())
@pytest.mark.parametrize("n_test_case", [1, 2, 3])
def test_univariate_loss_expected_zero(n_test_case, metric_func_name):
    # Test cases where the expected loss is zero for perfect forecast.

    metric_class = LOSS_RESULTS[metric_func_name]["class"]
    metric_func = LOSS_RESULTS[metric_func_name]["func"]

    y_true = Y_TEST_CASES[f"test_case_{n_test_case}"]["test"]
    y_train = Y_TEST_CASES[f"test_case_{n_test_case}"]["train"]

    # Setting test case of perfect forecast and benchmark
    true_loss = 0
    y_pred = y_true
    y_pred_benchmark = y_true

    function_loss, class_loss = _call_metrics(
        metric_func, metric_class, y_true, y_pred, y_train, y_pred_benchmark
    )

    # Assertion for functions
    assert np.isclose(function_loss, true_loss), " ".join(
        [
            f"Loss function {metric_func.__name__} returned {function_loss}",
            f"loss, but {true_loss} loss expected",
        ]
    )
    # Assertion for classes
    assert np.isclose(class_loss, true_loss), " ".join(
        [
            f"Loss function {metric_class.name} returned {class_loss}",
            f"loss, but {true_loss} loss expected",
        ]
    )


@pytest.mark.parametrize("metric_func_name", LOSS_RESULTS.keys())
@pytest.mark.parametrize("n_test_case", [1, 2, 3])
def test_univariate_loss_against_expected_value(n_test_case, metric_func_name):
    metric_class = LOSS_RESULTS[metric_func_name]["class"]
    metric_func = LOSS_RESULTS[metric_func_name]["func"]
    true_loss = LOSS_RESULTS[metric_func_name][f"test_case_{n_test_case}"]
    y_true = Y_TEST_CASES[f"test_case_{n_test_case}"]["test"]
    y_train = Y_TEST_CASES[f"test_case_{n_test_case}"]["train"]

    # Use last value as naive forecast to test function
    y_pred = np.concatenate([y_train, y_true])[23:35]

    # Just using this nonsensical approach to generate  benchmark for testing
    y_pred_benchmark = 0.6 * y_pred

    function_loss, class_loss = _call_metrics(
        metric_func, metric_class, y_true, y_pred, y_train, y_pred_benchmark
    )

    # Assertion for functions
    assert np.isclose(function_loss, true_loss), " ".join(
        [
            f"Loss function {metric_func.__name__} returned {function_loss}",
            f"loss, but {true_loss} loss expected",
        ]
    )
    # Assertion for classes
    assert np.isclose(class_loss, true_loss), " ".join(
        [
            f"Loss function {metric_class.name} returned {class_loss}",
            f"loss, but {true_loss} loss expected",
        ]
    )


@pytest.mark.parametrize("metric_func_name", LOSS_RESULTS.keys())
@pytest.mark.parametrize("random_state", RANDOM_STATES)
def test_univariate_metric_function_class_equality(metric_func_name, random_state):
    metric_class = LOSS_RESULTS[metric_func_name]["class"]
    metric_func = LOSS_RESULTS[metric_func_name]["func"]

    y = _make_series(n_timepoints=75, random_state=random_state)
    y_train, y_true = y.iloc[:50], y.iloc[50:]
    y_pred = y.shift(1).iloc[50:]
    y_pred_benchmark = y.rolling(2).mean().iloc[50:]

    function_loss, class_loss = _call_metrics(
        metric_func, metric_class, y_true, y_pred, y_train, y_pred_benchmark
    )

    # Assertion for functions and class having same result
    assert np.isclose(function_loss, class_loss), " ".join(
        [
            "Expected loss function and class to return equal values,",
            f"but loss function {metric_func.__name__} returned {function_loss}",
            f"and {metric_class.name} returned {class_loss}.",
        ]
    )


@pytest.mark.parametrize("random_state", RANDOM_STATES)
@pytest.mark.parametrize("metric_func_name", LOSS_RESULTS.keys())
def test_univariate_function_output_type(metric_func_name, random_state):
    metric_func = LOSS_RESULTS[metric_func_name]["func"]
    y = _make_series(n_timepoints=75, random_state=random_state)
    y_train, y_true = y.iloc[:50], y.iloc[50:]
    y_pred = y.shift(1).iloc[50:]
    y_pred_benchmark = y.rolling(2).mean().iloc[50:]

    function_loss = metric_func(
        y_true, y_pred, y_train=y_train, y_pred_benchmark=y_pred_benchmark
    )

    is_num = is_numeric_dtype(function_loss)
    is_scalar = np.isscalar(function_loss)
    assert is_num and is_scalar, " ".join(
        ["Loss function with univariate input should return scalar number"]
    )


@pytest.mark.parametrize("metric_func_name", LOSS_RESULTS.keys())
def test_y_true_y_pred_inconsistent_n_outputs_raises_error(metric_func_name):
    metric_func = LOSS_RESULTS[metric_func_name]["func"]
    y = _make_series(n_timepoints=75, random_state=RANDOM_STATES[0])
    y_train, y_true = y.iloc[:50], y.iloc[50:]
    y_true = y_true.values  # Convert to flat NumPy array
    y_pred = y.shift(1).iloc[50:]
    y_pred = np.expand_dims(y_pred.values, 1)  # convert to 1d NumPy array
    y_pred = np.hstack([y_pred, y_pred])
    y_pred_benchmark = y.rolling(2).mean().iloc[50:]

    # Test input types
    with pytest.raises(
        ValueError, match="y_true and y_pred have different number of output"
    ):
        metric_func(y_true, y_pred, y_train=y_train, y_pred_benchmark=y_pred_benchmark)


@pytest.mark.parametrize("metric_func_name", LOSS_RESULTS.keys())
def test_y_true_y_pred_inconsistent_n_timepoints_raises_error(metric_func_name):
    metric_func = LOSS_RESULTS[metric_func_name]["func"]
    y = _make_series(n_timepoints=75, random_state=RANDOM_STATES[0])
    y_train, y_true = y.iloc[:50], y.iloc[50:]
    y_pred = y.shift(1).iloc[40:]  # y_pred has more obs
    y_pred_benchmark = y.rolling(2).mean().iloc[50:]

    # Test input types
    with pytest.raises(
        ValueError, match="Found input variables with inconsistent numbers of samples"
    ):
        metric_func(y_true, y_pred, y_train=y_train, y_pred_benchmark=y_pred_benchmark)


@pytest.mark.parametrize("metric_func_name", LOSS_RESULTS.keys())
def test_y_true_y_pred_inconsistent_n_variables_raises_error(metric_func_name):
    metric_func = LOSS_RESULTS[metric_func_name]["func"]
    y = _make_series(n_timepoints=75, random_state=RANDOM_STATES[0])
    y_train, y_true = y.iloc[:50], y.iloc[50:]
    y_true = y_true.values  # will pass as NumPy array
    y_pred = y.shift(1).iloc[50:]
    y_pred = y_pred.to_frame()
    y_pred["Second Series"] = y.shift(1).iloc[50:]
    y_pred = y_pred.values
    y_pred_benchmark = y.rolling(2).mean().iloc[50:]

    # Test input types
    with pytest.raises(
        ValueError, match="y_true and y_pred have different number of output"
    ):
        metric_func(y_true, y_pred, y_train=y_train, y_pred_benchmark=y_pred_benchmark)
