"""Test for hidalgo segmentation."""

import numpy as np
import pytest
from sklearn.utils.validation import check_random_state

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.hidalgo import Hidalgo

# generate dataset
np.random.seed(10002)
X = np.zeros((10, 6))

# half the points from one generating regime
for j in range(1):
    X[:5, j] = np.random.normal(0, 3, 5)

# the other half from another
for j in range(3):
    X[5:, j] = np.random.normal(2, 1, 5)


@pytest.mark.skipif(
    not run_test_for_class(Hidalgo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_X():
    """Test if innput data is of expected dimension and type."""
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert len(np.shape(X)) == 2, "X should be a two-dimensional numpy array"


@pytest.mark.skipif(
    not run_test_for_class(Hidalgo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_get_neighbourhood_params():
    """Test for neighbourhood parameter generation."""
    model = Hidalgo(K=2, n_iter=10, burn_in=0.5, sampling_rate=2, seed=1)
    (
        N_actual,
        mu_actual,
        Iin_actual,
        Iout_actual,
        Iout_count_actual,
        Iout_track_actual,
    ) = model._get_neighbourhood_params(X)

    N_expected = 10
    mu_expected = np.array(
        [
            1.464722,
            5.40974013,
            2.464722,
            4.78175011,
            1.17460641,
            1.95890874,
            1.05863706,
            1.06313386,
            1.85040636,
            1.28065266,
        ]
    )
    Iin_expected = np.array(
        [
            2,
            4,
            7,
            3,
            9,
            4,
            0,
            4,
            7,
            1,
            4,
            9,
            0,
            9,
            3,
            8,
            6,
            9,
            8,
            5,
            9,
            5,
            8,
            9,
            5,
            6,
            7,
            5,
            7,
            4,
        ]
    )
    Iout_expected = np.array(
        [
            2,
            4,
            3,
            0,
            1,
            4,
            0,
            1,
            2,
            3,
            9,
            6,
            7,
            8,
            9,
            5,
            8,
            0,
            2,
            8,
            9,
            5,
            6,
            7,
            1,
            3,
            4,
            5,
            6,
            7,
        ]
    )
    Iout_count_expected = np.array([2, 1, 1, 2, 5, 4, 2, 4, 3, 6])
    Iout_track_expected = np.array([0, 2, 3, 4, 6, 11, 15, 17, 21, 24])

    assert np.allclose(N_actual, N_expected)
    assert np.allclose(mu_actual, mu_expected)
    assert np.allclose(Iin_actual, Iin_expected)
    assert np.allclose(Iout_actual, Iout_expected)
    assert np.allclose(Iout_count_actual, Iout_count_expected)
    assert np.allclose(Iout_track_actual, Iout_track_expected)


@pytest.mark.skipif(
    not run_test_for_class(Hidalgo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_initialise_params():
    """Test for initialise parameters."""
    model = Hidalgo(K=2, n_iter=10, burn_in=0.5, sampling_rate=2, seed=1)
    N = 10
    mu = np.array(
        [
            1.464722,
            5.40974013,
            2.464722,
            4.78175011,
            1.17460641,
            1.95890874,
            1.05863706,
            1.06313386,
            1.85040636,
            1.28065266,
        ]
    )
    Iin = np.array(
        [
            2,
            4,
            7,
            3,
            9,
            4,
            0,
            4,
            7,
            1,
            4,
            9,
            0,
            9,
            3,
            8,
            6,
            9,
            8,
            5,
            9,
            5,
            8,
            9,
            5,
            6,
            7,
            5,
            7,
            4,
        ]
    )
    _rng = check_random_state(model.seed)

    (
        V_actual,
        NN_actual,
        a1_actual,
        b1_actual,
        c1_actual,
        Z_actual,
        f1_actual,
        N_in_actual,
    ) = model._initialise_params(N, mu, Iin, _rng)

    V_expected = [2.7142554736925324, 3.6367957651001124]
    NN_expected = [3, 7]
    a1_expected = np.array([4.0, 8.0])
    b1_expected = np.array([3.71425547, 4.63679577])
    c1_expected = np.array([4.0, 8.0])
    Z_expected = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 0])
    f1_expected = np.array([15.0, 17.0])
    N_in_expected = 14

    assert np.allclose(V_actual, V_expected)
    assert np.allclose(NN_actual, NN_expected)
    assert np.allclose(a1_actual, a1_expected)
    assert np.allclose(b1_actual, b1_expected)
    assert np.allclose(c1_actual, c1_expected)
    assert np.allclose(Z_actual, Z_expected)
    assert np.allclose(f1_actual, f1_expected)
    assert N_in_actual == N_in_expected


@pytest.mark.skipif(
    not run_test_for_class(Hidalgo),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gibbs():
    """Test _transform method including gibbs sampling iterations."""
    expected = [
        [
            0.72269469,
            3.20993546,
            0.59599049,
            0.40400951,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -17.9650666,
            -4.50125762,
        ],
        [
            0.85025968,
            2.32993642,
            0.80166206,
            0.19833794,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            -19.21897121,
            -7.78773922,
        ],
    ]

    model = Hidalgo(K=2, n_iter=10, burn_in=0.5, sampling_rate=2, seed=1)
    _rng = check_random_state(model.seed)

    N, mu, Iin, Iout, Iout_count, Iout_track = model._get_neighbourhood_params(X)
    V, NN, a1, b1, c1, Z, f1, N_in = model._initialise_params(N, mu, Iin, _rng)

    sampling = model._gibbs_sampling(
        N,
        mu,
        Iin,
        Iout,
        Iout_count,
        Iout_track,
        V,
        NN,
        a1,
        b1,
        c1,
        Z,
        f1,
        N_in,
        _rng,
    )
    sampling = np.reshape(sampling, (10, 17))
    actual = sampling[[6, 8],]

    assert np.allclose(actual, expected)
