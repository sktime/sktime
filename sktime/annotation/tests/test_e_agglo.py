# -*- coding: utf-8 -*-
"""Tests for EAGGLO."""

__author__ = ["KatieBuc"]

import numpy as np
import pandas as pd

from sktime.annotation.e_agglo import EAGGLO


def test_fit_default_params_univariate():
    """Test data."""
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])

    cluster_expected = [0, 0, 1, 1]
    fit_expected = [104.77424, 134.51387, 186.92586, -31.15431]

    model = EAGGLO()
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster_
    fit_actual = fitted_model.fit_

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_fit_other_params_univariate():
    """Test data."""
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])

    cluster_expected = [0, 0, 1, 1]
    fit_expected = [1182.754, 1772.526, -295.421]

    model = EAGGLO(member=np.array([0, 0, 1, 2]), alpha=2)
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster_
    fit_actual = fitted_model.fit_

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_fit_default_params_multivariate():
    """Test data."""
    X = pd.DataFrame(
        [-6.475593, -6.709440, 4.892682, 4.748014, 5.476172],
        [-6.501258, -7.629093, -7.167619, -8.180040, -7.340993],
    )

    cluster_expected = [0, 0, 1, 1, 1]
    fit_expected = [118.57864, 132.67679, 156.54321, 208.60813, -33.52631]

    model = EAGGLO()
    fitted_model = model._fit(X)

    cluster_actual = fitted_model.cluster_
    fit_actual = fitted_model.fit_

    assert np.allclose(cluster_actual, cluster_expected)
    assert np.allclose(fit_actual, fit_expected)


def test_penalty():
    """Test data."""
    pass
