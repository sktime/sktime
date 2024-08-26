import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_almost_equal
from scipy import linalg

from sktime.tests.test_switch import run_test_module_changed

from ..unscented import AdditiveUnscentedKalmanFilter, cholupdate, qr


def build_unscented_filter(cls):
    """Instantiate the Unscented Kalman Filter"""
    # build transition functions
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    if cls == AdditiveUnscentedKalmanFilter:
        f = lambda x: A.dot(x)  # noqa: E731
        g = lambda x: C.dot(x)  # noqa: E731
    else:
        raise ValueError(f"How do I make transition functions for {cls}?")

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])

    Q = np.eye(2) * 2
    R = 0.5

    # build filter
    kf = cls(f, g, Q, R, x, P, random_state=0)

    return kf


def check_unscented_prediction(method, mu_true, sigma_true):
    """Check output of a method against true mean and covariances"""
    Z = ma.array([0, 1, 2, 3], mask=[True, False, False, False])
    (mu_est, sigma_est) = method(Z)
    mu_est, sigma_est = mu_est[1:], sigma_est[1:]

    assert_array_almost_equal(mu_true, mu_est, decimal=8)
    assert_array_almost_equal(sigma_true, sigma_est, decimal=8)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_additive_sample():
    kf = build_unscented_filter(AdditiveUnscentedKalmanFilter)
    (x, z) = kf.sample(100)

    assert x.shape == (100, 2)
    assert z.shape == (100, 1)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_additive_filter():
    # true unscented mean, covariance, as calculated by a MATLAB ukf_predict1
    # and ukf_update1 available from
    # http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.3563758389014, 0.929530201358681]
    mu_true[1] = [4.39153258609087, 1.15148930112108]
    mu_true[2] = [6.71906243585852, 1.52810614139809]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [
        [2.09738255033572, 1.51577181208044],
        [1.51577181208044, 2.91778523489926],
    ]
    sigma_true[1] = [
        [3.62532578216869, 3.14443733560774],
        [3.14443733560774, 4.65898912348032],
    ]
    sigma_true[2] = [
        [4.39024658597909, 3.90194406652556],
        [3.90194406652556, 5.40957304471631],
    ]

    check_unscented_prediction(
        build_unscented_filter(AdditiveUnscentedKalmanFilter).filter,
        mu_true,
        sigma_true,
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_additive_filter_update():
    kf = build_unscented_filter(AdditiveUnscentedKalmanFilter)
    Z = ma.array([0, 1, 2, 3], mask=[True, False, False, False])

    mu_filt, sigma_filt = kf.filter(Z)
    mu_filt2, sigma_filt2 = np.zeros(mu_filt.shape), np.zeros(sigma_filt.shape)
    for t in range(mu_filt.shape[0] - 1):
        if t == 0:
            mu_filt2[t] = mu_filt[t]
            sigma_filt2[t] = sigma_filt[t]
        mu_filt2[t + 1], sigma_filt2[t + 1] = kf.filter_update(
            mu_filt2[t], sigma_filt2[t], Z[t + 1]
        )

    assert_array_almost_equal(mu_filt, mu_filt2)
    assert_array_almost_equal(sigma_filt, sigma_filt2)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_additive_smoother():
    # true unscented mean, covariance, as calculated by a MATLAB urts_smooth1
    # available in http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.92725011499923, 1.63582509399207]
    mu_true[1] = [4.87447429622188, 1.64678689063005]
    mu_true[2] = [6.71906243585852, 1.52810614139809]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [
        [0.99379975649288, 0.21601451308325],
        [0.21601451308325, 1.25274857496361],
    ]
    sigma_true[1] = [
        [1.570868803779, 1.03741785934372],
        [1.03741785934372, 2.49806235789009],
    ]
    sigma_true[2] = [
        [4.39024658597909, 3.90194406652556],
        [3.90194406652556, 5.40957304471631],
    ]

    check_unscented_prediction(
        build_unscented_filter(AdditiveUnscentedKalmanFilter).smooth,
        mu_true,
        sigma_true,
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_cholupdate():
    M = np.array([[1, 0.2], [0.2, 0.8]])
    x = np.array([[0.3, 0.5], [0.01, 0.09]])
    w = -0.01

    R1 = linalg.cholesky(
        M
        + np.sign(w) * np.abs(w) * np.outer(x[0], x[0])
        + np.sign(w) * np.abs(w) * np.outer(x[1], x[1])
    )

    R2 = cholupdate(linalg.cholesky(M), x, w)

    assert_array_almost_equal(R1, R2)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_qr():
    A = np.array([[1, 0.2, 1], [0.2, 0.8, 2]]).T
    R = qr(A)
    assert R.shape == (2, 2)

    assert_array_almost_equal(R.T.dot(R), A.T.dot(A))
