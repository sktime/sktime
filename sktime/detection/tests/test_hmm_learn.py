"""Tests for hmmlearn wrapper annotation estimator."""

__author__ = ["miraep8", "klam-data", "pyyim", "mgorlin"]

import pytest
from numpy import array_equal

from sktime.detection.datagen import piecewise_normal, piecewise_poisson
from sktime.detection.hmm_learn import GMMHMM, GaussianHMM, PoissonHMM
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(GaussianHMM),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_GaussianHMM_wrapper():
    """Verify that the wrapped GaussianHMM estimator agrees with hmmlearn."""
    # moved all potential soft dependency import inside the test:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM

    data = piecewise_normal(
        means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ).reshape((-1, 1))
    hmmlearn_model = _GaussianHMM(n_components=3, random_state=7)
    sktime_model = GaussianHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.transform(X=data).values.flatten()
    assert array_equal(hmmlearn_predict, sktime_predict)


@pytest.mark.skipif(
    not run_test_for_class(GMMHMM),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_GMMHMM_wrapper():
    """Verify that the wrapped GMMHMM estimator agrees with hmmlearn."""
    # moved all potential soft dependency import inside the test:
    from hmmlearn.hmm import GMMHMM as _GMMHMM

    data = piecewise_normal(
        means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ).reshape((-1, 1))
    hmmlearn_model = _GMMHMM(n_components=3, random_state=7)
    sktime_model = GMMHMM(n_components=3, random_state=7)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.transform(X=data).values.flatten()
    assert array_equal(hmmlearn_predict, sktime_predict)


@pytest.mark.skipif(
    not run_test_for_class(PoissonHMM),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_PoissonHMM_wrapper():
    """Verify that the wrapped PoissonHMM estimator agrees with hmmlearn."""
    # moved all potential soft dependency import inside the test:
    from hmmlearn.hmm import PoissonHMM as _PoissonHMM

    data = piecewise_poisson(
        lambdas=[1, 2, 3], lengths=[2, 4, 8], random_state=42
    ).reshape((-1, 1))
    hmmlearn_model = _PoissonHMM(n_components=3, random_state=42)
    sktime_model = PoissonHMM(n_components=3, random_state=42)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.transform(X=data).values.flatten()
    assert array_equal(hmmlearn_predict, sktime_predict)
