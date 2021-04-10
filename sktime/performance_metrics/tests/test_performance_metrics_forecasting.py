#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Tomasz Chodakowski", "Ryan Kuhns"]

import pytest
import inspect
import numpy as np
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
    # relative_loss,
    # mean_asymmetric_error,
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
)
from sktime.performance_metrics.tests._config import (
    RANDOM_SEEDS,
    # TEST_YS,
    # TEST_YS_ZERO,
    Y_TEST_CASES,
)


# Dictionary mapping functions to the true loss values
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
    "_root_mean_squared_scaled_error": {
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
        "test_case_1": 16.2067453,
        "test_case_2": 17.0960482,
        "test_case_3": 108.3333333,
        "func": mean_absolute_percentage_error,
        "class": MeanAbsolutePercentageError(symmetric=True),
    },
    "symmetric_median_absolute_percentage_error": {
        "test_case_1": 17.2915592,
        "test_case_2": 15.3232867,
        "test_case_3": 150.0,
        "func": median_absolute_percentage_error,
        "class": MedianAbsolutePercentageError(symmetric=True),
    },
    "mean_absolute_percentage_error": {
        "test_case_1": 16.4263602,
        "test_case_2": 16.9569684,
        "test_case_3": 112.59e15,
        "func": mean_absolute_percentage_error,
        "class": MeanAbsolutePercentageError(),
    },
    "median_absolute_percentage_error": {
        "test_case_1": 17.2003523,
        "test_case_2": 15.2189132,
        "test_case_3": 100.0,
        "func": median_absolute_percentage_error,
        "class": MedianAbsolutePercentageError(),
    },
    "mean_squared_percentage_error": {
        "test_case_1": 320.34230364470864,
        "test_case_2": 342.74868218036704,
        "test_case_3": 5.070602400912918e34,
        "func": mean_squared_percentage_error,
        "class": MeanSquaredPercentageError(),
    },
    "median_squared_percentage_error": {
        "test_case_1": 295.89708748632586,
        "test_case_2": 231.72298452886963,
        "test_case_3": 10000.0,
        "func": median_squared_percentage_error,
        "class": MedianSquaredPercentageError(),
    },
    "root_mean_squared_percentage_error": {
        "test_case_1": 17.8981089,
        "test_case_2": 18.513473,
        "test_case_3": 225.18e15,
        "func": mean_squared_percentage_error,
        "class": MeanSquaredPercentageError(square_root=True),
    },
    "root_median_squared_percentage_error": {
        "test_case_1": 17.2016594,
        "test_case_2": 15.22245,
        "test_case_3": 100.0,
        "func": median_squared_percentage_error,
        "class": MedianSquaredPercentageError(square_root=True),
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
}


@pytest.mark.parametrize("loss_func_name", LOSS_RESULTS.keys())
@pytest.mark.parametrize("n_test_case", [1, 2, 3])
def test_univariate_loss_zero_error(n_test_case, loss_func_name):
    loss_class = LOSS_RESULTS[loss_func_name]["class"]
    loss_func = LOSS_RESULTS[loss_func_name]["func"]
    true_loss = 0
    y_true = Y_TEST_CASES[f"test_case_{n_test_case}"]["test"]
    y_train = Y_TEST_CASES[f"test_case_{n_test_case}"]["train"]
    # Testing case of perfect forecaster
    y_pred = y_true
    # Also perfect benchmark
    y_pred_benchmark = y_true

    loss_func_args = inspect.getfullargspec(loss_func).args
    if "y_train" in loss_func_args:
        function_loss = loss_func(y_true, y_pred, y_train)
        class_loss = loss_class.fn(y_true, y_pred, y_train)
    elif "y_pred_benchmark" in loss_func_args:
        function_loss = loss_func(y_true, y_pred, y_pred_benchmark)
        class_loss = loss_class.fn(y_true, y_pred, y_pred_benchmark)
    else:
        function_loss = loss_func(y_true, y_pred)
        class_loss = loss_class.fn(y_true, y_pred)

    # Assertion for functions
    assert np.isclose(function_loss, true_loss), " ".join(
        [
            f"Loss function {loss_func.__name__} returned {function_loss}",
            f"loss, but {true_loss} loss expected",
        ]
    )
    # Assertion for classes
    assert np.isclose(class_loss, true_loss), " ".join(
        [
            f"Loss function {loss_class.name} returned {class_loss}",
            f"loss, but {true_loss} loss expected",
        ]
    )


@pytest.mark.parametrize("loss_func_name", LOSS_RESULTS.keys())
@pytest.mark.parametrize("n_test_case", [1, 2, 3])
def test_univariate_loss_value(n_test_case, loss_func_name):
    loss_class = LOSS_RESULTS[loss_func_name]["class"]
    loss_func = LOSS_RESULTS[loss_func_name]["func"]
    true_loss = LOSS_RESULTS[loss_func_name][f"test_case_{n_test_case}"]
    y_true = Y_TEST_CASES[f"test_case_{n_test_case}"]["test"]
    y_train = Y_TEST_CASES[f"test_case_{n_test_case}"]["train"]
    # Use last value as naive forecast to test function
    y_pred = np.concatenate([y_train, y_true])[23:35]
    # Just using this nonsensical approach to generate  benchmark for testing
    y_pred_benchmark = 0.6 * y_pred

    class_attrs = loss_class.get_params()
    loss_func_args = inspect.getfullargspec(loss_func).args
    if "y_train" in loss_func_args:
        if len(class_attrs) > 0:
            function_loss = loss_func(y_true, y_pred, y_train, **class_attrs)
            class_loss = loss_class.fn(y_true, y_pred, y_train, **class_attrs)
        else:
            function_loss = loss_func(y_true, y_pred, y_train)
            class_loss = loss_class.fn(y_true, y_pred, y_train)
    elif "y_pred_benchmark" in loss_func_args:
        if len(class_attrs) > 0:
            function_loss = loss_func(y_true, y_pred, y_pred_benchmark, **class_attrs)
            class_loss = loss_class.fn(y_true, y_pred, y_pred_benchmark, **class_attrs)
        else:
            function_loss = loss_func(y_true, y_pred, y_pred_benchmark)
            class_loss = loss_class.fn(y_true, y_pred, y_pred_benchmark)
    else:
        if len(class_attrs) > 0:
            function_loss = loss_func(y_true, y_pred, **class_attrs)
            class_loss = loss_class.fn(y_true, y_pred, **class_attrs)
        else:
            function_loss = loss_func(y_true, y_pred)
            class_loss = loss_class.fn(y_true, y_pred)

    # Assertion for functions
    assert np.isclose(function_loss, true_loss), " ".join(
        [
            f"Loss function {loss_func.__name__} returned {function_loss}",
            f"loss, but {true_loss} loss expected",
        ]
    )
    # Assertion for classes
    assert np.isclose(class_loss, true_loss), " ".join(
        [
            f"Loss function {loss_class.name} returned {class_loss}",
            f"loss, but {true_loss} loss expected",
        ]
    )


@pytest.mark.parametrize("loss_func_name", LOSS_RESULTS.keys())
def test_univariate_function_class_equality(loss_func_name):
    loss_class = LOSS_RESULTS[loss_func_name]["class"]
    loss_func = LOSS_RESULTS[loss_func_name]["func"]

    for random_seed in RANDOM_SEEDS:
        y = _make_series(n_timepoints=75, random_state=random_seed)
        y_train, y_true = y.iloc[:50], y.iloc[50:]
        y_pred = y.shift(1).iloc[50:]
        y_pred_benchmark = y.rolling(2).mean().iloc[50:]

        class_attrs = loss_class.get_params()
        loss_func_args = inspect.getfullargspec(loss_func).args
        if "y_train" in loss_func_args:
            if len(class_attrs) > 0:
                function_loss = loss_func(y_true, y_pred, y_train, **class_attrs)
                class_loss = loss_class.fn(y_true, y_pred, y_train, **class_attrs)
            else:
                function_loss = loss_func(y_true, y_pred, y_train)
                class_loss = loss_class.fn(y_true, y_pred, y_train)
        elif "y_pred_benchmark" in loss_func_args:
            if len(class_attrs) > 0:
                function_loss = loss_func(
                    y_true, y_pred, y_pred_benchmark, **class_attrs
                )
                class_loss = loss_class.fn(
                    y_true, y_pred, y_pred_benchmark, **class_attrs
                )
            else:
                function_loss = loss_func(y_true, y_pred, y_pred_benchmark)
                class_loss = loss_class.fn(y_true, y_pred, y_pred_benchmark)
        else:
            if len(class_attrs) > 0:
                function_loss = loss_func(y_true, y_pred, **class_attrs)
                class_loss = loss_class.fn(y_true, y_pred, **class_attrs)
            else:
                function_loss = loss_func(y_true, y_pred)
                class_loss = loss_class.fn(y_true, y_pred)

        # Assertion for functions and class having same result
        assert np.isclose(function_loss, class_loss), " ".join(
            [
                "Expected loss function and class to return equal values,",
                f"but loss function {loss_func.__name__} returned {function_loss}",
                f"and {loss_class.name} returned {class_loss}.",
            ]
        )


@pytest.mark.parametrize("loss_func_name", LOSS_RESULTS.keys())
def test_univariate_function_output_type(loss_func_name):
    loss_func = LOSS_RESULTS[loss_func_name]["func"]
    for random_seed in RANDOM_SEEDS:
        y = _make_series(n_timepoints=75, random_state=random_seed)
        y_train, y_true = y.iloc[:50], y.iloc[50:]
        y_pred = y.shift(1).iloc[50:]
        y_pred_benchmark = y.rolling(2).mean().iloc[50:]

        loss_func_args = inspect.getfullargspec(loss_func).args

        if "y_train" in loss_func_args:
            function_loss = loss_func(y_true, y_pred, y_train)
        elif "y_pred_benchmark" in loss_func_args:
            function_loss = loss_func(y_true, y_pred, y_pred_benchmark)
        else:
            function_loss = loss_func(y_true, y_pred)

        is_num = is_numeric_dtype(function_loss)
        is_scalar = np.isscalar(function_loss)
        assert is_num and is_scalar, " ".join(
            ["Loss function with univariate input should return scalar number"]
        )


# @pytest.mark.parametrize("loss_func", LOSS_RESULTS.keys())
# def test_y_true_input_type(loss_func):
#     y = _make_series(n_timepoints=75, random_state=RANDOM_SEEDS[0])
#     y_train, y_true = y.iloc[:50], y.iloc[50:]
#     y_pred = y.shift(1).iloc[50:]
#     y_pred_benchmark = y.rolling(2).mean().iloc[50:]

#     loss_func_args = inspect.getfullargspec(loss_func).args

#     # create list version of each potential inputs
#     y_true_list = y_true.tolist()

#     # Test input types
#     with pytest.raises(TypeError):
#         if "y_train" in loss_func_args:
#             loss_func(y_true_list, y_pred, y_train)
#         elif "y_pred_benchmark" in loss_func_args:
#             loss_func(y_true_list, y_pred, y_pred_benchmark)
#         else:
#             loss_func(y_true_list, y_pred)


# @pytest.mark.parametrize("loss_func", LOSS_RESULTS.keys())
# def test_y_pred_input_type(loss_func):
#     y = _make_series(n_timepoints=75, random_state=RANDOM_SEEDS[0])
#     y_train, y_true = y.iloc[:50], y.iloc[50:]
#     y_pred = y.shift(1).iloc[50:]
#     y_pred_benchmark = y.rolling(2).mean().iloc[50:]

#     loss_func_args = inspect.getfullargspec(loss_func).args

#     # create list version of each potential inputs
#     y_pred_list = y_pred.tolist()

#     # Test input types
#     with pytest.raises(TypeError):
#         if "y_train" in loss_func_args:
#             loss_func(y_true, y_pred_list, y_train)
#         elif "y_pred_benchmark" in loss_func_args:
#             loss_func(y_true, y_pred_list, y_pred_benchmark)
#         else:
#            loss_func(y_true, y_pred_list)


# Only need to run test of y_train and y_pred_benchmark on the functions
# that accept those parameters
# @pytest.mark.parametrize(
#     "loss_func",
#     [
#         mean_absolute_scaled_error,
#         median_absolute_scaled_error,
#         mean_squared_scaled_error,
#         median_squared_scaled_error,
#         mean_relative_absolute_error,
#         median_relative_absolute_error,
#         geometric_mean_relative_absolute_error,
#         geometric_mean_relative_squared_error,
#     ],
# )
# def test_y_train_y_pred_benchmark_input_type(loss_func):
#     y = _make_series(n_timepoints=75, random_state=RANDOM_SEEDS[0])
#     y_train, y_true = y.iloc[:50], y.iloc[50:]
#     y_pred = y.shift(1).iloc[50:]
#     y_pred_benchmark = y.rolling(2).mean().iloc[50:]

#     loss_func_args = inspect.getfullargspec(loss_func).args

#     # create list version of each potential inputs
#     y_train_list = y_train.tolist()
#     y_pred_brenchmark_list = y_pred_benchmark.tolist()

#     # Test input types
#     with pytest.raises(TypeError):
#         if "y_train" in loss_func_args:
#             loss_func(y_true, y_pred, y_train_list)
#         elif "y_pred_benchmark" in loss_func_args:
#             loss_func(y_true, y_pred, y_pred_brenchmark_list)


@pytest.mark.parametrize("loss_func_name", LOSS_RESULTS.keys())
def test_y_true_y_pred_ndim(loss_func_name):
    loss_func = LOSS_RESULTS[loss_func_name]["func"]
    y = _make_series(n_timepoints=75, random_state=RANDOM_SEEDS[0])
    y_train, y_true = y.iloc[:50], y.iloc[50:]
    y_true = y_true.values  # Convert to flat NumPy array
    y_pred = y.shift(1).iloc[50:]
    y_pred = np.expand_dims(y_pred.values, 1)  # convert to 1d NumPy array
    y_pred = np.hstack([y_pred, y_pred])
    y_pred_benchmark = y.rolling(2).mean().iloc[50:]

    loss_func_args = inspect.getfullargspec(loss_func).args

    # Test input types
    with pytest.raises(ValueError):
        if "y_train" in loss_func_args:
            loss_func(y_true, y_pred, y_train)
        elif "y_pred_benchmark" in loss_func_args:
            loss_func(y_true, y_pred, y_pred_benchmark)
        else:
            loss_func(y_true, y_pred)


@pytest.mark.parametrize("loss_func_name", LOSS_RESULTS.keys())
def test_y_true_y_pred_equal_number_of_obs(loss_func_name):
    loss_func = LOSS_RESULTS[loss_func_name]["func"]
    y = _make_series(n_timepoints=75, random_state=RANDOM_SEEDS[0])
    y_train, y_true = y.iloc[:50], y.iloc[50:]
    y_pred = y.shift(1).iloc[40:]  # y_pred has more obs
    y_pred_benchmark = y.rolling(2).mean().iloc[50:]

    loss_func_args = inspect.getfullargspec(loss_func).args

    # Test input types
    with pytest.raises(ValueError):
        if "y_train" in loss_func_args:
            loss_func(y_true, y_pred, y_train)
        elif "y_pred_benchmark" in loss_func_args:
            loss_func(y_true, y_pred, y_pred_benchmark)
        else:
            loss_func(y_true, y_pred)


@pytest.mark.parametrize("loss_func_name", LOSS_RESULTS.keys())
def test_y_true_y_pred_equal_number_of_series(loss_func_name):
    loss_func = LOSS_RESULTS[loss_func_name]["func"]
    y = _make_series(n_timepoints=75, random_state=RANDOM_SEEDS[0])
    y_train, y_true = y.iloc[:50], y.iloc[50:]
    y_true = y_true.values  # will pass as NumPy array
    y_pred = y.shift(1).iloc[50:]
    y_pred = y_pred.to_frame()
    y_pred["Second Series"] = y.shift(1).iloc[50:]
    y_pred = y_pred.values
    y_pred_benchmark = y.rolling(2).mean().iloc[50:]

    loss_func_args = inspect.getfullargspec(loss_func).args

    # Test input types
    with pytest.raises(ValueError):
        if "y_train" in loss_func_args:
            loss_func(y_true, y_pred, y_train)
        elif "y_pred_benchmark" in loss_func_args:
            loss_func(y_true, y_pred, y_pred_benchmark)
        else:
            loss_func(y_true, y_pred)
