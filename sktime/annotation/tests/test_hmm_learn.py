# -*- coding: utf-8 -*-
"""Tests for hmmlearn wrapper annotation estimator."""

__author__ = ["miraep8"]

from hmmlearn.hmm import GaussianHMM as _GaussianHMM
from numpy import array_equal

from sktime.annotation.datagen import piecewise_normal
from sktime.annotation.hmm_learn import GaussianHMM


def test_wrapper_agrees_with_package():
    """Verify that the wrapped estimator agrees with the original package."""
    data = piecewise_normal(
        means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ).reshape((-1, 1))
    hmmlearn_model = _GaussianHMM(n_components=3, random_state=7)
    sktime_model = GaussianHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.predict(X=data)
    # will remove
    assert array_equal(hmmlearn_predict, sktime_predict)
