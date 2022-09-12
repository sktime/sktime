# -*- coding: utf-8 -*-
"""Tests for hmmlearn wrapper annotation estimator."""

__author__ = ["miraep8"]

import pytest
from numpy import array_equal

from sktime.annotation.datagen import piecewise_normal
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("hmmlearn", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_wrapper_agrees_with_package():
    """Verify that the wrapped estimator agrees with the original package."""
    # moved all potential soft dependency import inside the test:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM

    from sktime.annotation.hmm_learn import GaussianHMM

    data = piecewise_normal(means=[2, 4, 1], lengths=[10, 35, 40], random_state=7)
    hmmlearn_model = _GaussianHMM(n_components=3, random_state=7)
    sktime_model = GaussianHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.predict(X=data)
    assert array_equal(hmmlearn_predict, sktime_predict)
