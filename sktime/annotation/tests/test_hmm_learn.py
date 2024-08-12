"""Tests for hmmlearn wrapper annotation estimator."""

__author__ = ["miraep8", "klam-data", "pyyim", "mgorlin"]

import pytest
from numpy import array_equal

from sktime.annotation.datagen import piecewise_normal, piecewise_poisson
from sktime.utils.dependencies import _check_soft_dependencies


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
    assert array_equal(hmmlearn_predict, sktime_predict)


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
    assert array_equal(hmmlearn_predict, sktime_predict)


@pytest.mark.skipif(
    not _check_soft_dependencies("hmmlearn", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_PoissonHMM_wrapper():
    """Verify that the wrapped PoissonHMM estimator agrees with hmmlearn."""
    # moved all potential soft dependency import inside the test:
    from hmmlearn.hmm import PoissonHMM as _PoissonHMM

    from sktime.annotation.hmm_learn import PoissonHMM

    data = piecewise_poisson(
        lambdas=[1, 2, 3], lengths=[2, 4, 8], random_state=42
    ).reshape((-1, 1))
    hmmlearn_model = _PoissonHMM(n_components=3, random_state=42)
    sktime_model = PoissonHMM(n_components=3, random_state=42)
    hmmlearn_model.fit(X=data)
    sktime_model.fit(X=data)
    hmmlearn_predict = hmmlearn_model.predict(X=data)
    sktime_predict = sktime_model.predict(X=data)
    assert array_equal(hmmlearn_predict, sktime_predict)
