import numpy as np
import scipy.special as sp
import scipy.stats as st

from sktime.detection._skchange.utils.numba.stats import digamma, kurtosis, log_gamma, trigamma


def test_numba_kurtosis():
    seed = 523
    n_samples = 100
    p = 5
    t_dof = 5.0

    np.random.seed(seed)

    # Spd covariance matrix:
    random_nudge = np.random.randn(p).reshape(-1, 1)
    cov = np.eye(p) + 0.5 * random_nudge @ random_nudge.T

    mean = np.arange(p) * (-1 * np.ones(p)).cumprod()

    mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=t_dof)
    mv_t_samples = mv_t_dist.rvs(n_samples)

    sample_medians = np.median(mv_t_samples, axis=0)
    centered_samples = mv_t_samples - sample_medians

    # Test numba kurtosis:
    numba_kurtosis_val = kurtosis(centered_samples)
    scipy_kurtosis_val = st.kurtosis(centered_samples, fisher=True, bias=True, axis=0)

    assert np.all(np.isfinite(numba_kurtosis_val)), "Numba kurtosis should be finite."
    assert np.all(np.isfinite(scipy_kurtosis_val)), "Scipy kurtosis should be finite."
    (
        np.testing.assert_allclose(numba_kurtosis_val, scipy_kurtosis_val),
        "Kurtosis is off.",
    )


def test_numba_digamma():
    # Simpler test for digamma:
    np.random.seed(5125)
    random_vals = np.random.rand(100) * 10.0 + 0.5
    numba_digamma_vals = np.array([digamma(val) for val in random_vals])
    scipy_digamma_vals = sp.digamma(random_vals)
    np.testing.assert_allclose(numba_digamma_vals, scipy_digamma_vals, atol=1e-6)


def test_numba_digamma_on_small_values_gives_nan():
    # Test that digamma is not evaluated for small values:
    np.random.seed(5125)
    random_vals = np.random.rand(100) * 0.5
    numba_digamma_vals = np.array([digamma(val) for val in random_vals])
    less_than_threshold = random_vals <= 1.0e-2

    assert less_than_threshold.sum() > 0, "No small values found for digamma test."
    assert np.isnan(
        numba_digamma_vals[less_than_threshold]
    ).all(), "Numba digamma should be NaN on small values (<= 1.0e-2)."
    assert np.isfinite(
        numba_digamma_vals[~less_than_threshold]
    ).all(), "Numba digamma should be finite on values > 1.0e-2."


def test_numba_trigamma():
    np.random.seed(3125)
    random_vals = np.random.rand(100) * 10.0 + 0.5
    numba_trigamma_vals = np.array([trigamma(val) for val in random_vals])
    scipy_trigamma_vals = sp.polygamma(1, random_vals)
    np.testing.assert_allclose(numba_trigamma_vals, scipy_trigamma_vals, atol=1e-4)


def test_numba_trigamma_on_small_values_gives_nan():
    # Test that trigamma is not evaluated for small values:
    np.random.seed(5125)
    random_vals = np.random.rand(100) * 0.5
    numba_trigamma_vals = np.array([trigamma(val) for val in random_vals])
    less_than_threshold = random_vals <= 1.0e-2

    assert less_than_threshold.sum() > 0, "No small values found for trigamma test."
    assert np.isnan(
        numba_trigamma_vals[less_than_threshold]
    ).all(), "Numba trigamma should be NaN on small values (<= 1.0e-2)."
    assert np.isfinite(
        numba_trigamma_vals[~less_than_threshold]
    ).all(), "Numba trigamma should be finite on values > 1.0e-2."


def test_numba_log_gamma():
    np.random.seed(3125)
    # Our numba implementation should not be evaluated below 1.0:
    random_vals = np.random.rand(100) * 10.0 + 1.0
    numba_log_gamma_vals = np.array([log_gamma(val) for val in random_vals])
    scipy_log_gamma_vals = sp.gammaln(random_vals)
    np.testing.assert_allclose(numba_log_gamma_vals, scipy_log_gamma_vals, atol=1.0e-3)


def test_numba_log_gamma_on_small_values_gives_nan():
    # Test that log_gamma is not evaluated for small values:
    np.random.seed(5125)
    random_vals = np.random.rand(100) * 0.5
    numba_log_gamma_vals = np.array([log_gamma(val) for val in random_vals])
    less_than_threshold = random_vals <= 1.0e-2

    assert less_than_threshold.sum() > 0, "No small values found for log_gamma test."
    assert np.isnan(
        numba_log_gamma_vals[less_than_threshold]
    ).all(), "Numba log_gamma should be NaN on small values (<= 1.0e-2)."
    assert np.isfinite(
        numba_log_gamma_vals[~less_than_threshold]
    ).all(), "Numba log_gamma should be finite on values > 1.0e-2."
