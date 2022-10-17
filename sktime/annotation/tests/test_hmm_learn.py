# -*- coding: utf-8 -*-
"""Tests for hmmlearn wrapper annotation estimator."""

__author__ = ["miraep8"]

import numpy as np
import pytest

from sktime.annotation.datagen import piecewise_normal
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("hmmlearn", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_GaussianHMM_wrapper():
    """Verify that the wrapped GaussianHMM estimator agrees with hmmlearn."""
    # moved all potential soft dependency import inside the test:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM

    from sktime.annotation.hmm_learn import GaussianHMM

    data = piecewise_normal(
        means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ).reshape((-1, 1))
    hmmlearn_model = _GaussianHMM(n_components=3, random_state=7)
    sktime_model = GaussianHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.predict(X=data)
    assert np.array_equal(hmmlearn_predict, sktime_predict)


@pytest.mark.skipif(
    not _check_soft_dependencies("hmmlearn", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_GMMHMM_wrapper():
    """Verify that the wrapped GMMHMM estimator agrees with hmmlearn."""
    # moved all potential soft dependency import inside the test:
    from hmmlearn.hmm import GMMHMM as _GMMHMM

    from sktime.annotation.hmm_learn import GMMHMM

    data = piecewise_normal(
        means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ).reshape((-1, 1))
    hmmlearn_model = _GMMHMM(n_components=3, random_state=7)
    sktime_model = GMMHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.predict(X=data)
    assert np.array_equal(hmmlearn_predict, sktime_predict)


@pytest.mark.skipif(
    not _check_soft_dependencies("hmmlearn", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_CategoricalHMM_wrapper():
    """Verify that the wrapped CategoricalHMM estimator agrees with hmmlearn."""
    # moved all potential soft dependency import inside the test:
    from hmmlearn.hmm import CategoricalHMM as _CategoricalHMM

    from sktime.annotation.hmm_learn import CategoricalHMM

    data = np.array([[0], [1], [2]]).reshape((-1, 1))
    hmmlearn_model = _CategoricalHMM(n_components=3, random_state=7)
    sktime_model = CategoricalHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.predict(X=data)
    assert np.array_equal(hmmlearn_predict, sktime_predict)
