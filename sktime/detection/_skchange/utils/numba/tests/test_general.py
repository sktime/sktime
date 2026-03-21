import numpy as np
import pytest

from ..general import compute_finite_difference_derivatives


def test_quadratic_derivative_accuracy():
    # y = t^2, y'' = 2
    ts = np.linspace(0, 1, 10)
    ys = ts**2
    deriv = compute_finite_difference_derivatives(ts, ys)
    second_derivative = compute_finite_difference_derivatives(ts, deriv)
    # Should be close to 2 everywhere
    assert np.allclose(second_derivative, 2.0, atol=1e-10)


def test_sine_derivative_accuracy():
    # y = sin(t), y' = cos(t)
    ts = np.linspace(0, 2 * np.pi, 100)
    ys = np.sin(ts)
    true_d = np.cos(ts)
    approx_d = compute_finite_difference_derivatives(ts, ys)
    # Ignore endpoints (less accurate)
    assert np.allclose(approx_d, true_d, atol=5.0e-3)


def test_raises_with_less_than_three_points():
    ts = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        compute_finite_difference_derivatives(ts, ys)


def test_convergence_order():
    # y = sin(t), y' = cos(t)
    max_errors = []
    step_sizes = []
    argmax_errors = []

    for n in [20, 40, 80, 160, 320, 640]:
        ts = np.linspace(0, 1, n)
        ys = np.sin(ts)
        approx_d = compute_finite_difference_derivatives(ts, ys)
        # Ignore endpoints
        abs_errors = np.abs(approx_d - np.cos(ts))
        argmax_errors.append(np.argmax(abs_errors))
        max_error = np.max(abs_errors)
        max_errors.append(max_error)
        step_sizes.append(ts[1] - ts[0])

    # Fit log-log slope
    slope, _ = np.polyfit(np.log(step_sizes), np.log(max_errors), 1)

    # Should be close to 2 (second order)
    assert 1.999 < slope < 2.001
