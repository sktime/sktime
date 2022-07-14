# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for (dunder) composition functionality attached to the base class."""

__author__ = ["fkiraly"]
__all__ = []

import pandas as pd
from sklearn.preprocessing import StandardScaler

from sktime.datatypes import get_examples
from sktime.transformations.compose import FeatureUnion, TransformerPipeline
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.subset import ColumnSelect
from sktime.transformations.series.summarize import SummaryTransformer
from sktime.transformations.series.theta import ThetaLinesTransformer
from sktime.utils._testing.deep_equals import deep_equals
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


def test_mul_sklearn_autoadapt():
    """Test auto-adapter for sklearn in mul."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t1 = ExponentTransformer(power=2)
    t2 = StandardScaler()
    t3 = ExponentTransformer(power=0.5)

    t123 = t1 * t2 * t3
    t123r = t1 * (t2 * t3)
    t123l = (t1 * t2) * t3

    assert isinstance(t123, TransformerPipeline)
    assert isinstance(t123r, TransformerPipeline)
    assert isinstance(t123l, TransformerPipeline)

    _assert_array_almost_equal(t123.fit_transform(X), t123l.fit_transform(X))
    _assert_array_almost_equal(t123r.fit_transform(X), t123l.fit_transform(X))


def test_missing_unequal_tag_inference():
    """Test that TransformerPipeline infers missing/unequal tags correctly."""
    t1 = ExponentTransformer() * PaddingTransformer() * ExponentTransformer()
    t2 = ExponentTransformer() * ExponentTransformer()
    t3 = Imputer() * ExponentTransformer()
    t4 = ExponentTransformer() * Imputer()

    assert t1.get_tag("capability:unequal_length")
    assert t1.get_tag("capability:unequal_length:removes")
    assert not t2.get_tag("capability:unequal_length:removes")
    assert t3.get_tag("handles-missing-data")
    assert t3.get_tag("capability:missing_values:removes")
    assert not t4.get_tag("handles-missing-data")
    assert not t4.get_tag("capability:missing_values:removes")


def test_featureunion_transform_cols():
    """Test FeatureUnion name and number of columns."""
    X = pd.DataFrame({"test1": [1, 2], "test2": [3, 4]})

    t1 = ExponentTransformer(power=2)
    t2 = ExponentTransformer(power=5)
    t3 = ExponentTransformer(power=3)

    t123 = t1 + t2 + t3

    Xt = t123.fit_transform(X)

    expected_cols = pd.Index(
        [
            "ExponentTransformer_1__test1",
            "ExponentTransformer_1__test2",
            "ExponentTransformer_2__test1",
            "ExponentTransformer_2__test2",
            "ExponentTransformer_3__test1",
            "ExponentTransformer_3__test2",
        ]
    )

    msg = (
        f"FeatureUnion creates incorrect column names for DataFrame output. "
        f"Expected: {expected_cols}, found: {Xt.columns}"
    )

    assert deep_equals(Xt.columns, expected_cols), msg


def test_sklearn_after_primitives():
    """Test that sklearn transformer after primitives is correctly applied."""
    t = SummaryTransformer() * StandardScaler()
    assert t.get_tag("scitype:transform-output") == "Primitives"

    X = get_examples("pd-multiindex")[0]
    X_out = t.fit_transform(X)
    X_summary = SummaryTransformer().fit_transform(X)

    assert deep_equals(X_out.index, X_summary.index)
    assert deep_equals(X_out.columns, X_summary.columns)
    # var_0 is the same for all three instances
    # so summary statistics are all the same, thus StandardScaler transforms to 0
    assert X_out.iloc[0, 0] > -0.01
    assert X_out.iloc[0, 0] < 0.01
    # var_1 has some variation between three instances
    # fix this to one value to tie the output to current behaviour
    assert X_out.iloc[0, 10] > -1.37
    assert X_out.iloc[0, 10] < -1.36


def test_pipeline_column_vectorization():
    """Test that pipelines vectorize properly over columns."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t = ColumnSelect([0, 1]) * ThetaLinesTransformer()

    X_theta = t.fit_transform(X)

    assert set(X_theta.columns) == set(["a__0", "a__2", "b__0", "b__2"])
