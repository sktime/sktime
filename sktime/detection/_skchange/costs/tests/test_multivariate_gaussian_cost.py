import numpy as np
import pytest
from scipy.stats import multivariate_normal

from sktime.detection._skchange.costs import MultivariateGaussianCost


def analytical_mv_ll_at_mle(X: np.ndarray):
    """
    Calculate the analytical multivariate log-likelihood at the MLE.

    Parameters
    ----------
    X : `np.ndarray`
        2D array.

    Returns
    -------
    ll : `float`
        Log-likelihood.
    """
    n = X.shape[0]
    p = X.shape[1]

    # MLE:
    sigma = np.cov(X, rowvar=False, ddof=0).reshape(p, p)

    # Log-likelihood:
    det_sign, logdet_sigma = np.linalg.slogdet(sigma)
    if det_sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")

    ll = 0.0
    ll = -((n * p) / 2) * np.log(2 * np.pi) - (n / 2) * logdet_sigma
    ll -= (1 / 2) * p * n

    return ll


@pytest.mark.parametrize(
    "n, p, seed",
    [
        (50, 1, 4125),
        (50, 3, 4125),
        (100, 1, 4125),
        (100, 3, 4125),
        (50, 1, 2125),
        (50, 3, 2125),
        (100, 1, 2125),
        (100, 3, 2125),
    ],
)
def test_mean_cov_cost(n: int, p: int, seed: int):
    """Test mean covariance cost."""
    # Generate data:
    np.random.seed(seed)
    X = np.random.randn(n, p)

    cost = MultivariateGaussianCost.create_test_instance()
    cost.fit(X)
    cost_value = cost.evaluate(np.array([[0, n]]))[0, 0]

    # Analytical cost using SciPy:
    mu_X = np.mean(X, axis=0)
    cov_X = np.cov(X, rowvar=False, ddof=0)
    mvn_dist = multivariate_normal(mean=mu_X, cov=cov_X)
    numerical_mle_ll = mvn_dist.logpdf(X).sum()
    numerical_cost = -2 * numerical_mle_ll

    # Analytical cost from theoretical formulae:
    analytical_mle_ll = analytical_mv_ll_at_mle(X)
    analytical_cost = -2 * analytical_mle_ll

    assert np.allclose(numerical_mle_ll, analytical_mle_ll)
    assert np.allclose(numerical_cost, analytical_cost)
    assert np.allclose(cost_value, analytical_cost)


@pytest.mark.parametrize(
    "mean, cov",
    [
        (0.0, np.array([[1, 0], [0, -1]])),  # Negative eigenvalue
        (0.0, np.array([[1, 2], [2, 1]])),  # Not positive definite
        (0.0, np.array([[1, 0.5], [0.5, 0]])),  # Zero eigenvalue
        (0.0, np.array([1.0, 2.0])),  # Wrong ndim
        (0.0, np.array([[1.0]])),  # Wrong shape (should be 2 x 2 if array input)
    ],
)
def test_invalid_fixed_covariance(mean, cov):
    """Test that invalid fixed covariance matrix raises errors."""
    X = np.random.randn(100, 2)
    cost = MultivariateGaussianCost(param=(mean, cov))
    with pytest.raises(ValueError):
        cost.fit(X)


def test_mean_cov_cost_raises_on_non_positive_definite():
    """Test mean covariance cost raises on non-positive definite."""
    X = np.array([[1, 2], [2, 1], [1, 2]])
    n = X.shape[0]
    cost = MultivariateGaussianCost.create_test_instance()
    cost.fit(X)

    with pytest.raises(RuntimeError):
        cost.evaluate(np.array([0, n]))

    # Check that the analytical function raises on the same input:
    with pytest.raises(ValueError):
        analytical_mv_ll_at_mle(X)


def test_mean_cov_cost_raises_on_single_observation():
    """Test mean covariance cost raises on single observation."""
    X = np.array([[1, 2]])
    n = X.shape[0]
    cost = MultivariateGaussianCost.create_test_instance()
    cost.fit(X)

    with pytest.raises(ValueError):
        cost.evaluate([0, n])
