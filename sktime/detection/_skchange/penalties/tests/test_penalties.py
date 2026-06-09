"""Tests for all penalties."""

import numpy as np
import pytest

from sktime.detection._skchange.penalties import (
    make_bic_penalty,
    make_chi2_penalty,
    make_linear_chi2_penalty,
    make_linear_penalty,
    make_mvcapa_penalty,
    make_nonlinear_chi2_penalty,
)


def test_make_bic_penalty():
    penalty = make_bic_penalty(2, 100)
    assert isinstance(penalty, float) or (
        isinstance(penalty, np.ndarray) and penalty.size == 1
    )
    assert penalty >= 0.0

    with pytest.raises(ValueError):
        make_bic_penalty(0, 100, 1)
    with pytest.raises(ValueError):
        make_bic_penalty(2, 0, 1)
    with pytest.raises(ValueError):
        make_bic_penalty(2, 100, -1)


def test_make_chi2_penalty():
    penalty = make_chi2_penalty(4, 20)
    assert isinstance(penalty, float) or (
        isinstance(penalty, np.ndarray) and penalty.size == 1
    )
    assert penalty >= 0.0

    with pytest.raises(ValueError):
        make_chi2_penalty(0, 100)
    with pytest.raises(ValueError):
        make_chi2_penalty(2, 0)


def test_make_linear_chi2_penalty():
    p = 4
    penalty = make_linear_chi2_penalty(3, 100, p)
    assert isinstance(penalty, np.ndarray)
    assert penalty.shape == (p,)
    assert np.all(penalty >= 0)
    assert np.all(np.diff(penalty) >= 0)

    with pytest.raises(ValueError):
        make_linear_chi2_penalty(0, 1, 1)
    with pytest.raises(ValueError):
        make_linear_chi2_penalty(1, 0, 1)
    with pytest.raises(ValueError):
        make_linear_chi2_penalty(1, 1, 0)


def test_make_linear_penalty():
    p = 10
    penalty = make_linear_penalty(10, 5, p)
    assert isinstance(penalty, np.ndarray)
    assert penalty.shape == (p,)
    assert np.all(penalty >= 0)
    assert np.all(np.diff(penalty) >= 0)

    with pytest.raises(ValueError):
        make_linear_penalty(-1, 1, 1)
    with pytest.raises(ValueError):
        make_linear_penalty(1, -1, 1)
    with pytest.raises(ValueError):
        make_linear_penalty(1, 1, 0)


def test_make_nonlinear_chi2_penalty():
    p = 100
    penalty = make_nonlinear_chi2_penalty(10, 200, p)
    assert isinstance(penalty, np.ndarray)
    assert penalty.shape == (p,)
    assert np.all(penalty >= 0)
    assert np.all(np.diff(penalty) >= 0)

    with pytest.raises(ValueError):
        make_linear_chi2_penalty(0, 1, 1)
    with pytest.raises(ValueError):
        make_linear_chi2_penalty(1, 0, 1)
    with pytest.raises(ValueError):
        make_linear_chi2_penalty(1, 1, 0)

    penalty_p1 = make_nonlinear_chi2_penalty(10, 200, 1)
    assert penalty_p1.size == 1


def test_make_mvcapa_penalty():
    p = 100
    penalty = make_mvcapa_penalty(10, 200, p)
    assert isinstance(penalty, np.ndarray)
    assert penalty.shape == (p,)
    assert np.all(penalty >= 0)
    assert np.all(np.diff(penalty) >= 0)

    with pytest.raises(ValueError):
        make_mvcapa_penalty(0, 1, 1)
    with pytest.raises(ValueError):
        make_mvcapa_penalty(1, 0, 1)
    with pytest.raises(ValueError):
        make_mvcapa_penalty(1, 1, 0)
