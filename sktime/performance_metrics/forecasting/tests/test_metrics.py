# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for some metrics."""
# currently this consists entirely of doctests from _classes and _functions
# since the numpy output print changes between versions

import numpy as np
import pandas as pd


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
    gmse = GeometricMeanSquaredError(multioutput="raw_values")
    assert np.allclose(gmse(y_true, y_pred), np.array([2.30997255e-11, 1.00000000e00]))
    rgmse = GeometricMeanSquaredError(multioutput="raw_values", square_root=True)
    assert np.allclose(rgmse(y_true, y_pred), np.array([4.80621738e-06, 1.00000000e00]))
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
        gmse(y_true, y_pred, multioutput="raw_values"),
        np.array([2.30997255e-11, 1.00000000e00]),
    )
    assert np.allclose(
        gmse(y_true, y_pred, multioutput="raw_values", square_root=True),
        np.array([4.80621738e-06, 1.00000000e00]),
    )
    assert np.allclose(gmse(y_true, y_pred, multioutput=[0.3, 0.7]), 0.7000000000069299)
    assert np.allclose(
        gmse(y_true, y_pred, multioutput=[0.3, 0.7], square_root=True),
        0.7000014418652152,
    )

    assert np.allclose(
        gmse(
            np.array([1, 2, 3]), np.array([6, 5, 4]), horizon_weight=np.array([7, 8, 9])
        ),
        6.185891035775025,
    )


def test_linex_class():
    """Doctest from MeanLinexError."""
    from sktime.performance_metrics.forecasting import MeanLinexError

    linex_error = MeanLinexError()
    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    assert np.allclose(linex_error(y_true, y_pred), 0.19802627763937575)
    linex_error = MeanLinexError(b=2)
    assert np.allclose(linex_error(y_true, y_pred), 0.3960525552787515)
    linex_error = MeanLinexError(a=-1)
    assert np.allclose(linex_error(y_true, y_pred), 0.2391800623225643)
    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    linex_error = MeanLinexError()
    assert np.allclose(linex_error(y_true, y_pred), 0.2700398392309829)
    linex_error = MeanLinexError(a=-1)
    assert np.allclose(linex_error(y_true, y_pred), 0.49660966225813563)
    linex_error = MeanLinexError(multioutput="raw_values")
    assert np.allclose(linex_error(y_true, y_pred), np.array([0.17220024, 0.36787944]))
    linex_error = MeanLinexError(multioutput=[0.3, 0.7])
    assert np.allclose(linex_error(y_true, y_pred), 0.30917568000716666)


def test_linex_function():
    """Doctest from mean_linex_error."""
    from sktime.performance_metrics.forecasting import mean_linex_error

    y_true = np.array([3, -0.5, 2, 7, 2])
    y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    assert np.allclose(mean_linex_error(y_true, y_pred), 0.19802627763937575)
    assert np.allclose(mean_linex_error(y_true, y_pred, b=2), 0.3960525552787515)
    assert np.allclose(mean_linex_error(y_true, y_pred, a=-1), 0.2391800623225643)
    y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    assert np.allclose(mean_linex_error(y_true, y_pred), 0.2700398392309829)
    assert np.allclose(mean_linex_error(y_true, y_pred, a=-1), 0.49660966225813563)
    assert np.allclose(
        mean_linex_error(y_true, y_pred, multioutput="raw_values"),
        np.array([0.17220024, 0.36787944]),
    )
    assert np.allclose(
        mean_linex_error(y_true, y_pred, multioutput=[0.3, 0.7]), 0.30917568000716666
    )


def test_make_scorer():
    """Test make_forecasting_scorer and the failure case in #4827."""
    import functools

    from sklearn.metrics import mean_squared_log_error

    from sktime.performance_metrics.forecasting import make_forecasting_scorer

    rmsle = functools.partial(mean_squared_log_error, squared=False)

    scorer = make_forecasting_scorer(rmsle, name="RMSLE")

    scorer.evaluate(pd.Series([1, 2, 3]), pd.Series([1, 2, 4]))
