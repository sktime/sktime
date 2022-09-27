# -*- coding: utf-8 -*-
"""Tests for E-Agglo (agglomerative clustering algorithm)."""

__author__ = ["KatieBuc"]

import numpy as np
import pandas as pd

from sktime.annotation.e_agglo import EAGGLO


def test_fit_default_params_univariate():
    """Test univariate data and defualt parameters.

    These numbers are generated from the original implemention
    in R, with the following code:

    set.seed(1234)
    X <- c(rnorm(15, mean = -6), 0, rnorm(16, mean = 6))
    X <- as.matrix(X[c(1,2,17,18)])
    ret = e.agglo(X)
    """
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])

    cluster_expected = [0, 0, 1, 1]
    fit_expected = [104.77424, 134.51387, 186.92586, -31.15431]

    model = EAGGLO()
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster
    fit_actual = fitted_model.gof

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_fit_other_params_univariate():
    """Test univariate data with alternative starting clusters."""
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])

    cluster_expected = [0, 0, 1, 1]
    fit_expected = [1182.754, 1772.526, -295.421]

    model = EAGGLO(member=np.array([0, 0, 1, 2]), alpha=2)
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster
    fit_actual = fitted_model.gof

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_fit_default_params_multivariate():
    """Test multivariate data with defauly parameters.

    These numbers are generated from the original implemention
    in R, with the following code:

    set.seed(1234)
    X <- c(rnorm(15, mean = -6), 0, rnorm(16, mean = 6))
    X <- as.matrix(cbind(X[c(1,2,17,18,19)],X[c(3,4,5,6,7)]))
    ret = e.agglo(X)
    """
    X = pd.DataFrame(
        [
            [-7.207, -4.916],
            [-5.723, -8.346],
            [5.890, -5.571],
            [5.489, -5.494],
            [5.089, -6.575],
        ]
    )

    cluster_expected = [0, 0, 1, 1, 1]
    fit_expected = [118.58235, 132.67919, 156.54531, 208.61512, -33.52743]

    model = EAGGLO()
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster
    fit_actual = fitted_model.gof

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_penalty1():
    """Test multivariate data with penalty function as string input."""
    X = pd.DataFrame(
        [
            [-7.207, -4.916],
            [-5.723, -8.346],
            [5.890, -5.571],
            [5.489, -5.494],
            [5.089, -6.575],
        ]
    )

    cluster_expected = [0, 0, 1, 1, 1]
    fit_expected = [112.58235, 127.67919, 152.54531, 205.61512, -35.52743]

    model = EAGGLO(penalty="penalty1")
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster
    fit_actual = fitted_model.gof

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_penalty2():
    """Test multivariate data with functional input as penalty."""
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])

    cluster_expected = [0, 0, 1, 1]
    fit_expected = [105.77424, 135.84720, 188.92586, -29.15431]

    model = EAGGLO(penalty=lambda x: np.mean(np.diff(np.sort(x))))
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster
    fit_actual = fitted_model.gof

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)
