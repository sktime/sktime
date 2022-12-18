# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for some metrics."""
# currently this consists entirely of doctests from _classes and _functions
# since the numpy output print changes between versions

import numpy as np


def test_gmse_class():
    """Doctest from GeometricMeanSquaredError."""
    from sktime.performance_metrics.forecasting import GeometricMeanSquaredError

    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    gmse = GeometricMeanSquaredError()

    assert np.allclose(gmse(y_true, y_pred), 2.80399089461488e-07)
    rgmse = GeometricMeanSquaredError(square_root=True)
    assert np.allclose(rgmse(y_true, y_pred), 0.000529527232030127)

    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    gmse = GeometricMeanSquaredError()
    assert np.allclose(gmse(y_true, y_pred), 0.5000000000115499)
    rgmse = GeometricMeanSquaredError(square_root=True)
    assert np.allclose(rgmse(y_true, y_pred), 0.5000024031086919)
    gmse = GeometricMeanSquaredError(multioutput='raw_values')
    assert np.allclose(gmse(y_true, y_pred), np.array([2.30997255e-11, 1.00000000e+00]))
    rgmse = GeometricMeanSquaredError(multioutput='raw_values', square_root=True)
    assert np.allclose(
        rgmse(y_true, y_pred), np.array([4.80621738e-06, 1.00000000e+00])
    )
    gmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7])
    assert np.allclose(gmse(y_true, y_pred), 0.7000000000069299)
    rgmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    assert np.allclose(rgmse(y_true, y_pred), 0.7000014418652152)


def test_gmse_function():
    """Doctest from geometric_mean_squared_error."""
    from sktime.performance_metrics.forecasting import geometric_mean_squared_error

    gmse = geometric_mean_squared_error
    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    assert np.allclose(gmse(y_true, y_pred), 2.80399089461488e-07)
    assert np.allclose(gmse(y_true, y_pred, square_root=True), 0.000529527232030127)
    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    assert np.allclose(gmse(y_true, y_pred), 0.5000000000115499)
    assert np.allclose(gmse(y_true, y_pred, square_root=True), 0.5000024031086919)
    assert np.allclose(
        gmse(y_true, y_pred, multioutput='raw_values'),
        np.array([2.30997255e-11, 1.00000000e+00]),
    )
    assert np.allclose(
        gmse(y_true, y_pred, multioutput='raw_values', square_root=True),
        np.array([4.80621738e-06, 1.00000000e+00]),
    )
    assert np.allclose(gmse(y_true, y_pred, multioutput=[0.3, 0.7]), 0.7000000000069299)
    assert np.allclose(
        gmse(y_true, y_pred, multioutput=[0.3, 0.7], square_root=True),
        0.7000014418652152,
    )
