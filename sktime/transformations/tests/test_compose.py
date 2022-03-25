# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for (dunder) composition functionality attached to the base class."""

__author__ = ["fkiraly"]
__all__ = []

import pandas as pd

from sktime.transformations.compose import FeatureUnion, TransformerPipeline
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal


def test_dunder_mul():
    """Test the mul dunder method."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t1 = ExponentTransformer(power=2)
    t2 = ExponentTransformer(power=5)
    t3 = ExponentTransformer(power=0.1)
    t4 = ExponentTransformer(power=1)

    t12 = t1 * t2
    t123 = t12 * t3
    t312 = t3 * t12
    t1234 = t123 * t4
    t1234_2 = t12 * (t3 * t4)

    assert isinstance(t12, TransformerPipeline)
    assert isinstance(t123, TransformerPipeline)
    assert isinstance(t312, TransformerPipeline)
    assert isinstance(t1234, TransformerPipeline)
    assert isinstance(t1234_2, TransformerPipeline)

    assert [x.power for x in t12.steps] == [2, 5]
    assert [x.power for x in t123.steps] == [2, 5, 0.1]
    assert [x.power for x in t312.steps] == [0.1, 2, 5]
    assert [x.power for x in t1234.steps] == [2, 5, 0.1, 1]
    assert [x.power for x in t1234_2.steps] == [2, 5, 0.1, 1]

    _assert_array_almost_equal(X, t123.fit_transform(X))
    _assert_array_almost_equal(X, t312.fit_transform(X))
    _assert_array_almost_equal(X, t1234.fit_transform(X))
    _assert_array_almost_equal(X, t1234_2.fit_transform(X))
    _assert_array_almost_equal(t12.fit_transform(X), t3.fit(X).inverse_transform(X))


def test_dunder_add():
    """Test the add dunder method."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t1 = ExponentTransformer(power=2)
    t2 = ExponentTransformer(power=5)
    t3 = ExponentTransformer(power=3)

    t12 = t1 + t2
    t123 = t12 + t3
    t123r = t1 + (t2 + t3)

    assert isinstance(t12, FeatureUnion)
    assert isinstance(t123, FeatureUnion)
    assert isinstance(t123r, FeatureUnion)

    assert [x.power for x in t12.transformer_list] == [2, 5]
    assert [x.power for x in t123.transformer_list] == [2, 5, 3]
    assert [x.power for x in t123r.transformer_list] == [2, 5, 3]

    _assert_array_almost_equal(t123r.fit_transform(X), t123.fit_transform(X))
