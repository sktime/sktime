import numpy as np
from scipy.special import digamma

from sktime.detection._skchange.change_detectors import MovingWindow
from sktime.detection._skchange.change_scores._multivariate_gaussian_score import (
    MultivariateGaussianScore,
    _half_integer_digamma,
)


def test_digamma():
    integer_vals = np.arange(1, 25)

    integer_vals = np.concatenate([integer_vals, 100 + integer_vals])

    manual_digamma = np.array(list(map(_half_integer_digamma, 2 * integer_vals)))
    scipy_digamma = digamma(integer_vals)

    np.testing.assert_allclose(manual_digamma, scipy_digamma)


def test_GaussianCovScore():
    np.random.seed(0)
    X_1 = np.random.normal(size=(100, 3), loc=[1.0, -0.2, 0.5], scale=[1.0, 0.5, 1.5])
    X_2 = np.random.normal(size=(100, 3), loc=[-1.0, 0.2, -0.5], scale=[4.0, 1.5, 2.8])

    X = np.concatenate([X_1, X_2, X_1], axis=0)
    cuts = np.array([[0, 50, 100], [0, 100, 200], [100, 200, 300], [0, 150, 300]])

    scores = MultivariateGaussianScore().fit(X).evaluate(cuts)

    assert scores.shape == (cuts.shape[0], 1)
    assert np.all(scores >= 0)


def test_scores_differ_with_Bartlett_correction():
    np.random.seed(123)
    X_1 = np.random.normal(size=(100, 3), loc=[1.0, -0.2, 0.5], scale=[1.0, 0.5, 1.5])
    X_2 = np.random.normal(size=(100, 3), loc=[-1.0, 0.2, -0.5], scale=[4.0, 1.5, 2.8])

    X = np.concatenate([X_1, X_2], axis=0)
    cuts = np.array([[0, 25, 50], [0, 50, 100], [50, 100, 150], [0, 100, 200]])

    raw_scores = (
        MultivariateGaussianScore(apply_bartlett_correction=False).fit(X).evaluate(cuts)
    )
    corrected_scores = (
        MultivariateGaussianScore(apply_bartlett_correction=True).fit(X).evaluate(cuts)
    )

    assert np.all(raw_scores > corrected_scores)


def test_non_fitted_GaussianCovScore_no_min_size():
    assert MultivariateGaussianScore().min_size is None


def test_mv_gaussian_score_on_MovingWindow():
    np.random.seed(0)
    X_1 = np.random.normal(size=(100, 3), loc=[1.0, -0.2, 0.5], scale=[1.0, 0.5, 1.5])
    X_2 = np.random.normal(size=(100, 3), loc=[-1.0, 0.2, -0.5], scale=[4.0, 1.5, 2.8])

    X = np.concatenate([X_1, X_2], axis=0)

    cost = MultivariateGaussianScore()

    change_detector = MovingWindow(change_score=cost, bandwidth=50, penalty=1.0)
    change_detector.fit(X)
    change_points = change_detector.predict(X)

    assert np.all(change_points["ilocs"].to_numpy() == np.array([100]))
