# -*- coding: utf-8 -*-
"""Tests for hmmlearn wrapper annotation estimator."""

__author__ = ["miraep8"]

from hmmlearn.hmm import GMMHMM as _GMMHMM
from hmmlearn.hmm import GaussianHMM as _GaussianHMM
from numpy import array_equal

from sktime.annotation.datagen import piecewise_normal
from sktime.annotation.hmm_learn import GMMHMM, GaussianHMM


def test_GaussianHMM_wrapper():
    """Verify that the wrapped GaussianHMM estimator agrees with hmmlearn."""
    data = piecewise_normal(
        means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ).reshape((-1, 1))
    hmmlearn_model = _GaussianHMM(n_components=3, random_state=7)
    sktime_model = GaussianHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.predict(X=data)
    assert array_equal(hmmlearn_predict, sktime_predict)


def test_GMMHMM_wrapper():
    """Verify that the wrapped GMMHMM estimator agrees with hmmlearn."""
    data = piecewise_normal(
        means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ).reshape((-1, 1))
    hmmlearn_model = _GMMHMM(n_components=3, random_state=7)
    sktime_model = GMMHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.predict(X=data)
    assert array_equal(hmmlearn_predict, sktime_predict)
