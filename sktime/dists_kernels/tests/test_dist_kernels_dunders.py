# -*- coding: utf-8 -*-
"""Tests for pairwise panel transformer dunders."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels.algebra import CombinedDistance
from sktime.dists_kernels.compose import PwTrafoPanelPipeline
from sktime.dists_kernels.edit_dist import EditDist
from sktime.utils._testing.panel import _make_panel_X

X1 = _make_panel_X(n_instances=5, n_columns=5, n_timepoints=5, random_state=1)
X2 = _make_panel_X(n_instances=6, n_columns=5, n_timepoints=5, random_state=2)


def test_mul_algebra_dunder():
    """Test multiplication dunder, algebraic case (two panel distances)."""
    t1 = EditDist()
    t2 = EditDist(distance="edr")
    t3 = EditDist(distance="erp")

    m1 = t1.transform(X1, X2)
    m2 = t2.transform(X1, X2)
    m3 = t3.transform(X1, X2)

    t12 = t1 * t2
    assert isinstance(t12, CombinedDistance)
    assert len(t12.pw_trafos) == 2
    assert t12.get_params()["operation"] == "*"

    m12 = t12.transform(X1, X2)
    assert np.allclose(m12, m1 * m2)

    t123 = t1 * t2 * t3
    assert isinstance(t123, CombinedDistance)
    assert len(t123.pw_trafos) == 3
    assert t123.get_params()["operation"] == "*"

    m123 = t123.transform(X1, X2)
    assert np.allclose(m123, m1 * m2 * m3)

    t123r = t1 * (t2 * t3)
    assert isinstance(t123r, CombinedDistance)
    assert len(t123r.pw_trafos) == 3
    assert t123r.get_params()["operation"] == "*"

    m123r = t123r.transform(X1, X2)
    assert np.allclose(m123r, m1 * m2 * m3)


def test_add_algebra_dunder():
    """Test addition dunder, algebraic case (two panel distances)."""
    t1 = EditDist()
    t2 = EditDist(distance="edr")
    t3 = EditDist(distance="erp")

    m1 = t1.transform(X1, X2)
    m2 = t2.transform(X1, X2)
    m3 = t3.transform(X1, X2)

    t12 = t1 + t2
    assert isinstance(t12, CombinedDistance)
    assert len(t12.pw_trafos) == 2
    assert t12.get_params()["operation"] == "+"

    m12 = t12.transform(X1, X2)
    assert np.allclose(m12, m1 + m2)

    t123 = t1 + t2 + t3
    assert isinstance(t123, CombinedDistance)
    assert len(t123.pw_trafos) == 3
    assert t123.get_params()["operation"] == "+"

    m123 = t123.transform(X1, X2)
    assert np.allclose(m123, m1 + m2 + m3)

    t123r = t1 + (t2 + t3)
    assert isinstance(t123r, CombinedDistance)
    assert len(t123r.pw_trafos) == 3
    assert t123r.get_params()["operation"] == "+"

    m123r = t123r.transform(X1, X2)
    assert np.allclose(m123r, m1 + m2 + m3)


def test_mixed_algebra_dunders():
    """Test mix of algebraic dunders."""
    t1 = EditDist()
    t2 = EditDist(distance="edr")
    t3 = EditDist(distance="erp")

    m1 = t1.transform(X1, X2)
    m2 = t2.transform(X1, X2)
    m3 = t3.transform(X1, X2)

    t123 = t1 * t2 + t3
    assert isinstance(t123, CombinedDistance)
    assert len(t123.pw_trafos) == 2
    assert t123.get_params()["operation"] == "+"
    t12 = t123.pw_trafos[0]
    assert isinstance(t12, CombinedDistance)
    assert len(t12.pw_trafos) == 2
    assert t12.get_params()["operation"] == "*"

    m123 = t123.transform(X1, X2)
    assert np.allclose(m123, m1 * m2 + m3)


def test_pw_trafo_pipeline_mul_dunder():
    """Tests creation of pairwise panel trafo pipeliens using mul dunder."""
    from sktime.transformations.series.exponent import ExponentTransformer

    t3 = EditDist()
    t1 = ExponentTransformer()
    t2 = ExponentTransformer(2)

    m3 = t3.transform(X1, X2)

    t23 = t2 * t3
    assert isinstance(t23, PwTrafoPanelPipeline)
    assert len(t23.transformers) == 1

    t123 = t1 * t2 * t3
    assert isinstance(t123, PwTrafoPanelPipeline)
    assert len(t23.transformers) == 2

    m123 = t123.transform(X1, X2)

    assert np.allclose(m123, m3)
