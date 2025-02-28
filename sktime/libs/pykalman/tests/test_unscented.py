import inspect

import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_almost_equal

from sktime.tests.test_switch import run_test_module_changed

from ..datasets import load_robot
from ..unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter

data = load_robot()


def build_unscented_filter(cls):
    """Instantiate the Unscented Kalman Filter"""
    # build transition functions
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    if cls == UnscentedKalmanFilter:
        f = lambda x, y: A.dot(x) + y  # noqa: E731
        g = lambda x, y: C.dot(x) + y  # noqa: E731
    elif cls == AdditiveUnscentedKalmanFilter:
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


def check_dims(n_dim_state, n_dim_obs, n_func_args, kf_cls, kwargs):
    kf = kf_cls(**kwargs)
    (
        transition_functions,
        observation_functions,
        transition_covariance,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
    ) = kf._initialize_parameters()

    assert (
        transition_functions.shape == (1,)
        if "transition_functions" not in kwargs
        else (len(kwargs["transition_functions"]),)
    )
    assert all(
        [
            len(inspect.getfullargspec(f).args) == n_func_args
            for f in transition_functions
        ]
    )
    assert transition_covariance.shape == (n_dim_state, n_dim_state)
    assert (
        observation_functions.shape == (1,)
        if "observation_functions" not in kwargs
        else (len(kwargs["observation_functions"]),)
    )
    assert all(
        [
            len(inspect.getfullargspec(f).args) == n_func_args
            for f in observation_functions
        ]
    )
    assert observation_covariance.shape == (n_dim_obs, n_dim_obs)
    assert initial_state_mean.shape == (n_dim_state,)
    assert initial_state_covariance.shape == (n_dim_state, n_dim_state)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_unscented_sample():
    kf = build_unscented_filter(UnscentedKalmanFilter)
    (x, z) = kf.sample(100)

    assert x.shape == (100, 2)
    assert z.shape == (100, 1)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_unscented_filter():
    # true unscented mean, covariance, as calculated by a MATLAB ukf_predict3
    # and ukf_update3 available from
    # http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.35637583900053, 0.92953020131845]
    mu_true[1] = [4.39153258583784, 1.15148930114305]
    mu_true[2] = [6.71906243764755, 1.52810614201467]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [
        [2.09738255033564, 1.51577181208054],
        [1.51577181208054, 2.91778523489934],
    ]
    sigma_true[1] = [
        [3.62532578216913, 3.14443733560803],
        [3.14443733560803, 4.65898912348045],
    ]
    sigma_true[2] = [
        [4.3902465859811, 3.90194406652627],
        [3.90194406652627, 5.40957304471697],
    ]

    check_unscented_prediction(
        build_unscented_filter(UnscentedKalmanFilter).filter, mu_true, sigma_true
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_unscented_filter_update():
    kf = build_unscented_filter(UnscentedKalmanFilter)
    Z = ma.array([0, 1, 2, 3], mask=[True, False, False, False])

    mu_filt, sigma_filt = kf.filter(Z)
    mu_filt2, sigma_filt2 = np.zeros(mu_filt.shape), np.zeros(sigma_filt.shape)
    for t in range(mu_filt.shape[0] - 1):
        if t == 0:
            mu_filt2[t] = mu_filt[0]
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
def test_unscented_smoother():
    # true unscented mean, covariance, as calculated by a MATLAB urts_smooth2
    # available in http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.92725011530645, 1.63582509442842]
    mu_true[1] = [4.87447429684622, 1.6467868915685]
    mu_true[2] = [6.71906243764755, 1.52810614201467]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [
        [0.993799756492982, 0.216014513083516],
        [0.216014513083516, 1.25274857496387],
    ]
    sigma_true[1] = [
        [1.57086880378025, 1.03741785934464],
        [1.03741785934464, 2.49806235789068],
    ]
    sigma_true[2] = [
        [4.3902465859811, 3.90194406652627],
        [3.90194406652627, 5.40957304471697],
    ]

    check_unscented_prediction(
        build_unscented_filter(UnscentedKalmanFilter).smooth, mu_true, sigma_true
    )


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
            mu_filt2[t] = mu_filt[0]
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
def test_unscented_initialize_parameters():
    check_dims(
        1,
        1,
        2,
        UnscentedKalmanFilter,
        {"transition_functions": [lambda x, y: x, lambda x, y: x]},
    )
    check_dims(3, 5, 2, UnscentedKalmanFilter, {"n_dim_state": 3, "n_dim_obs": 5})
    check_dims(1, 3, 2, UnscentedKalmanFilter, {"observation_covariance": np.eye(3)})
    check_dims(2, 1, 2, UnscentedKalmanFilter, {"initial_state_mean": np.zeros(2)})


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.pykalman"),
    reason="Execute tests for pykalman iff anything in the module has changed",
)
def test_additive_initialize_parameters():
    check_dims(
        1,
        1,
        1,
        AdditiveUnscentedKalmanFilter,
        {"transition_functions": [lambda x: x, lambda x: x]},
    )
    check_dims(
        3, 5, 1, AdditiveUnscentedKalmanFilter, {"n_dim_state": 3, "n_dim_obs": 5}
    )
    check_dims(
        1, 3, 1, AdditiveUnscentedKalmanFilter, {"observation_covariance": np.eye(3)}
    )
    check_dims(
        2, 1, 1, AdditiveUnscentedKalmanFilter, {"initial_state_mean": np.zeros(2)}
    )
