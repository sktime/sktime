# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for transformer composition functionality attached to the base class."""

__author__ = ["fkiraly", "aminmiral"]
__all__ = []

import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps
from sklearn.preprocessing import StandardScaler

from sktime.datasets import load_airline, load_unit_test
from sktime.datatypes import get_examples
from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.base import BaseTransformer
from sktime.transformations.bootstrap import STLBootstrapTransformer
from sktime.transformations.boxcox import LogTransformer
from sktime.transformations.compose import (
    FeatureUnion,
    GroupbyCategoryTransformer,
    InvertTransform,
    IxToX,
    OptionalPassthrough,
    TransformerPipeline,
)
from sktime.transformations.difference import Differencer
from sktime.transformations.exponent import ExponentTransformer
from sktime.transformations.impute import Imputer
from sktime.transformations.padder import PaddingTransformer
from sktime.transformations.subset import ColumnSelect
from sktime.transformations.summarize import SummaryTransformer
from sktime.transformations.theta import ThetaLinesTransformer
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils.deep_equals import deep_equals


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_add_sklearn_autoadapt():
    """Test the add dunder method, with sklearn coercion."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t1 = ExponentTransformer(power=2)
    t2 = StandardScaler()
    t3 = ExponentTransformer(power=3)

    t123 = t1 + t2 + t3
    t123r = t1 + (t2 + t3)
    t123l = (t1 + t2) + t3

    assert isinstance(t123, FeatureUnion)
    assert isinstance(t123r, FeatureUnion)
    assert isinstance(t123l, FeatureUnion)

    _assert_array_almost_equal(t123.fit_transform(X), t123l.fit_transform(X))
    _assert_array_almost_equal(t123r.fit_transform(X), t123l.fit_transform(X))


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_mul_sklearn_autoadapt():
    """Test the mul dunder method, with sklearn coercion."""
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


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_missing_unequal_tag_inference():
    """Test that TransformerPipeline infers missing/unequal tags correctly."""
    t1 = ExponentTransformer() * PaddingTransformer() * ExponentTransformer()
    t2 = ExponentTransformer() * ExponentTransformer()
    t3 = Imputer() * ExponentTransformer()
    t4 = ExponentTransformer() * Imputer()

    assert t1.get_tag("capability:unequal_length")
    assert t1.get_tag("capability:unequal_length:removes")
    assert not t2.get_tag("capability:unequal_length:removes")
    assert t3.get_tag("capability:missing_values")
    assert t3.get_tag("capability:missing_values:removes")
    assert not t4.get_tag("capability:missing_values")
    assert not t4.get_tag("capability:missing_values:removes")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
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


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_featureunion_primitives():
    """Test that FeatureUnion is correctly applied to primitives.

    Failure case of bug #6077.
    """
    X, _ = load_unit_test(split="train", return_X_y=True)

    fu = SummaryTransformer() + SummaryTransformer()
    Xt = fu.fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert len(Xt) == len(X)
    assert Xt.shape[1] == 2 * 9  # 9-feature summary statistics
    assert Xt.columns[0] == "SummaryTransformer_1__mean"  # unique naming


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_sklearn_after_primitives():
    """Test that sklearn transformer after primitives is correctly applied."""
    t = SummaryTransformer() * StandardScaler()
    assert t.get_tag("scitype:transform-output") == "Primitives"

    X = get_examples("pd-multiindex")[0]
    X_out = t.fit_transform(X)
    X_summary = SummaryTransformer().fit_transform(X)

    assert (X_out.index == X_summary.index).all()
    assert deep_equals(X_out.columns, X_summary.columns)
    # var_0 is the same for all three instances
    # so summary statistics are all the same, thus StandardScaler transforms to 0
    assert X_out.iloc[0, 0] > -0.01
    assert X_out.iloc[0, 0] < 0.01
    # var_1 has some variation between three instances
    # fix this to one value to tie the output to current behaviour
    assert X_out.iloc[0, 10] > -1.37
    assert X_out.iloc[0, 10] < -1.36


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_pipeline_column_vectorization():
    """Test that pipelines vectorize properly over columns."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t = ColumnSelect([0, 1]) * ThetaLinesTransformer()

    X_theta = t.fit_transform(X)

    assert set(X_theta.columns) == {"a__0", "a__2", "b__0", "b__2"}


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_pipeline_inverse():
    """Tests that inverse composition works, with inverse skips.

    Also see #3084.
    """
    X = load_airline()
    t = LogTransformer() * Imputer()

    # LogTransformer has inverse_transform, and does not skip inverse transform
    # therefore, pipeline should also not skip inverse transform, and have capability
    assert t.get_tag("capability:inverse_transform")
    assert not t.get_tag("skip-inverse-transform")

    t.fit(X)
    Xt = t.transform(X)
    Xtt = t.inverse_transform(Xt)

    _assert_array_almost_equal(X, Xtt)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_subset_getitem():
    """Test subsetting using the [ ] dunder, __getitem__."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    t = ThetaLinesTransformer()

    t_before = t["a"]
    t_before_with_colon = t[["a", "b"], :]
    t_after_with_colon = t[:, ["a__0", "a__2"]]
    t_both = t[["a", "b"], ["b__0", "b__2", "c__0", "c__2"]]
    t_none = t[:, :]

    assert isinstance(t_before, TransformerPipeline)
    assert isinstance(t_after_with_colon, TransformerPipeline)
    assert isinstance(t_before_with_colon, TransformerPipeline)
    assert isinstance(t_both, TransformerPipeline)
    assert isinstance(t_none, ThetaLinesTransformer)

    X_theta = t.fit_transform(X)

    _assert_array_almost_equal(t_before.fit_transform(X), X_theta[["a__0", "a__2"]])
    _assert_array_almost_equal(
        t_after_with_colon.fit_transform(X), X_theta[["a__0", "a__2"]]
    )
    _assert_array_almost_equal(
        t_before_with_colon.fit_transform(X), X_theta[["a__0", "a__2", "b__0", "b__2"]]
    )
    _assert_array_almost_equal(t_both.fit_transform(X), X_theta[["b__0", "b__2"]])
    _assert_array_almost_equal(t_none.fit_transform(X), X_theta)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_dunder_invert():
    """Test the invert dunder method, for wrapping in OptionalPassthrough."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t = ExponentTransformer(power=3)

    t_inv = ~t

    assert isinstance(t_inv, InvertTransform)
    assert isinstance(t_inv.get_params()["transformer"], ExponentTransformer)

    _assert_array_almost_equal(
        t_inv.fit_transform(X), ExponentTransformer(1 / 3).fit_transform(X)
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_dunder_neg():
    """Test the neg dunder method, for wrapping in OptionalPassthrough."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t = ExponentTransformer(power=2)

    tp = -t

    assert isinstance(tp, OptionalPassthrough)
    assert not tp.get_params()["passthrough"]
    assert isinstance(tp.get_params()["transformer"], ExponentTransformer)

    _assert_array_almost_equal(tp.fit_transform(X), X)


@pytest.mark.skipif(
    not _check_estimator_deps(STLBootstrapTransformer, severity="none"),
    reason="skip test if required soft dependency for statsmodels not available",
)
def test_input_output_series_panel_chain():
    """Test that series-to-panel can be chained with series-to-series trafos.

    Failure case of #5624.
    """
    from sktime.datasets import load_airline
    from sktime.transformations.impute import Imputer

    X = load_airline()
    bootstrap_trafo = STLBootstrapTransformer(4, sp=4) * Imputer(method="nearest")

    assert bootstrap_trafo.get_tag("scitype:transform-input") == "Series"
    assert bootstrap_trafo.get_tag("scitype:transform-output") == "Panel"

    Xt = bootstrap_trafo.fit_transform(X)
    assert isinstance(Xt, pd.DataFrame)
    assert isinstance(Xt.index, pd.MultiIndex)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_requires_tags_trafopipe():
    """Test correct handling of requires_X tag, failure case in ."""
    from sktime.transformations.compose import TransformerPipeline, YtoX
    from sktime.transformations.fourier import FourierFeatures

    # data with no exogenous features
    X = load_airline()

    # create a pipeline with Fourier features and ARIMA
    pipe = TransformerPipeline(
        steps=[
            YtoX(),
            FourierFeatures(
                sp_list=[24, 24 * 7],
                fourier_terms_list=[10, 5],
                keep_original_columns=True,
            ),
        ]
    )

    assert not pipe.get_tag("requires_X")
    # should not requires X as input, because YtoX does not

    assert pipe.get_tag("requires_y")
    # should require y as input, because YtoX does

    pipe.fit_transform(X=None, y=X)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_ixtox():
    index = pd.MultiIndex.from_product(
        [["X", "Y"], ["A", "B", "C"], [1, 2, 3]],
        names=["level_0", "level_1", "level_2"],
    )
    X = pd.DataFrame(index=index)

    ixtox = IxToX(level="__all_but_time")
    assert ixtox.fit_transform(X).columns.tolist() == ["level_0", "level_1"]

    ixtox = IxToX(level="__all")
    assert ixtox.fit_transform(X).columns.tolist() == ["level_0", "level_1", "level_2"]

    ixtox = IxToX(level=None)
    assert ixtox.fit_transform(X).columns.tolist() == ["level_2"]

    ixtox = IxToX(level=["level_0", "level_2"])
    assert ixtox.fit_transform(X).columns.tolist() == ["level_0", "level_2"]

    ixtox = IxToX(level=-1)
    assert ixtox.fit_transform(X).columns.tolist() == ["level_2"]


class _MeanCategorizer(BaseTransformer):
    """Test-only transformer: category = str(round(mean)) of each instance.

    Used in place of ADICVTransformer to give tests full, deterministic
    control over which category each panel instance is assigned to.
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "capability:missing_values": True,
    }

    def _transform(self, X, y=None):
        category = str(round(X.mean().mean()))
        return pd.DataFrame({"category": [category]})


def _make_category_panel(instance_means, n_timepoints=3):
    """Build a panel where each instance is a constant series at instance_means[i]."""
    frames = []
    for i, mean in enumerate(instance_means):
        idx = pd.MultiIndex.from_product(
            [[f"inst_{i}"], range(n_timepoints)], names=["instance", "time"]
        )
        frames.append(pd.DataFrame({"c0": [float(mean)] * n_timepoints}, index=idx))
    return pd.concat(frames)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_groupby_category_transformer_fit_transform():
    """Test that each category gets its own fitted transformer, incl. fallback."""
    X = _make_category_panel([1.0, 1.0, 100.0, 100.0, 500.0])

    t = GroupbyCategoryTransformer(
        transformers={"1": Differencer(), "100": ExponentTransformer(power=2)},
        categorizer=_MeanCategorizer(),
        fallback_transformer=Differencer(),
    )
    t.fit(X)

    assert set(t.transformers_.keys()) == {"1", "100", "500"}
    assert isinstance(t.transformers_["1"], Differencer)
    assert isinstance(t.transformers_["100"], ExponentTransformer)
    assert isinstance(t.transformers_["500"], Differencer)  # fallback was used

    Xt = t.transform(X)
    assert Xt.index.equals(X.index)

    # category "100" was routed through ExponentTransformer(power=2)
    high_rows = Xt.loc[["inst_2", "inst_3"]]
    assert (high_rows["c0"] == 100.0**2).all()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_groupby_category_transformer_transform_new_data():
    """Test transform re-categorizes the passed X, instead of replaying fit groups.

    Regression test: transform must call the fitted categorizer on the data it
    is actually given, not reuse the instance grouping computed during fit.
    Otherwise, transforming a subset of the fitted instances silently produces
    a wrong or malformed result instead of matching the input exactly.
    """
    X = _make_category_panel([1.0, 1.0, 100.0, 100.0])

    t = GroupbyCategoryTransformer(
        transformers={"1": Differencer(), "100": ExponentTransformer(power=2)},
        categorizer=_MeanCategorizer(),
    )
    t.fit(X)

    X_new = X.loc[["inst_2", "inst_3"]]
    Xt_new = t.transform(X_new)

    assert Xt_new.index.equals(X_new.index)
    assert (Xt_new["c0"] == 100.0**2).all()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_groupby_category_transformer_unknown_category_raises():
    """Test that a category with no transformer and no fallback raises."""
    X = _make_category_panel([1.0, 500.0])

    t = GroupbyCategoryTransformer(
        transformers={"1": Differencer()},
        categorizer=_MeanCategorizer(),
    )
    with pytest.raises(ValueError, match="500"):
        t.fit(X)
