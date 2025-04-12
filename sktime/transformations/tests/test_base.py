# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for base class conversion and vectorization functionality.

Each test covers a "decision path" in the base class boilerplate,     with a focus on
frequently breaking paths in base class refactor and bugfixing. The path taken depends
on tags of a given transformer, and input data type. Concrete transformer classes from
sktime are imported to cover     different combinations of transformer tags. Transformer
scenarios cover different combinations of input data types.
"""

__author__ = ["fkiraly"]
__all__ = []

from inspect import isclass

import pandas as pd
import pytest

from sktime.datatypes import check_is_scitype, get_examples, mtype_to_scitype
from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.compose import FitInTransform
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.transformations.panel.tsfresh import (
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor,
)
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.summarize import SummaryTransformer
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.scenarios_transformers import (
    TransformerFitTransformHierarchicalMultivariate,
    TransformerFitTransformHierarchicalUnivariate,
    TransformerFitTransformPanelUnivariate,
    TransformerFitTransformPanelUnivariateWithClassYOnlyFit,
    TransformerFitTransformSeriesMultivariate,
    TransformerFitTransformSeriesUnivariate,
)
from sktime.utils._testing.series import _make_series
from sktime.utils.dependencies import _check_estimator_deps
from sktime.utils.parallel import _get_parallel_test_fixtures

# other scenarios that might be needed later in development:
# TransformerFitTransformPanelUnivariateWithClassY,

# list of parallelization backends to test
BACKENDS = _get_parallel_test_fixtures("config")


def inner_X_scitypes(est):
    """Return list of scitypes supported by class est, as list of str."""
    if isclass(est):
        X_inner_mtype = est.get_class_tag("X_inner_mtype")
    else:
        X_inner_mtype = est.get_tag("X_inner_mtype")
    X_inner_scitypes = mtype_to_scitype(
        X_inner_mtype, return_unique=True, coerce_to_list=True
    )
    return X_inner_scitypes


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_series_in_series_out_supported():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = False
        "X_inner_mtype" supports "Series

    X input to fit/transform has Series scitype
    X output from fit/transform should be Series
    """
    # one example for a transformer which supports Series internally
    cls = BoxCoxTransformer
    est = cls.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Series", return_metadata=True)
    assert valid, "fit.transform does not return a Series when given a Series"
    # todo: possibly, add mtype check, use metadata return


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_series_in_series_out_supported_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = True
        "X_inner_mtype" supports "Series"

    X input to fit/transform has Series scitype
    X output from fit/transform should be Series
    """
    # one example for a transformer which supports Series internally
    cls = ExponentTransformer
    est = cls.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert est.get_class_tag("fit_is_empty")
    assert est.get_class_tag("scitype:transform-input") == "Series"
    assert est.get_class_tag("scitype:transform-output") == "Series"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Series", return_metadata=True)
    assert valid, "fit.transform does not return a Series when given a Series"
    # todo: possibly, add mtype check, use metadata return


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_series_in_series_out_not_supported_but_panel():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = False
        "X_inner_mtype" does not support "Series" but does support "Panel"
            i.e., none of the mtypes in the list is "Series" but some are "Panel"

    X input to fit/transform has Series scitype
    X output from fit/transform should be Series
    """
    # one example for a transformer which supports Panel internally but not Series
    cls = PaddingTransformer
    est = cls.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Panel" in inner_X_scitypes(est)
    assert "Series" not in inner_X_scitypes(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Series", return_metadata=True)
    assert valid, "fit.transform does not return a Series when given a Series"
    # todo: possibly, add mtype check, use metadata return


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_panel_in_panel_out_supported():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = False
        "X_inner_mtype" supports "Panel"

    X input to fit/transform has Panel scitype
    X output from fit/transform should be Panel
    """
    # one example for a transformer which supports Panel internally
    cls = PaddingTransformer
    est = cls.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Panel" in inner_X_scitypes(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformPanelUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Panel", return_metadata=True)
    assert valid, "fit.transform does not return a Panel when given a Panel"
    # todo: possibly, add mtype check, use metadata return


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_panel_in_panel_out_not_supported_but_series(backend):
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = False
        "X_inner_mtype" supports "Series" but not "Panel" and not "Hierarchical"

    X input to fit/transform has Panel scitype
    X output from fit/transform should be Panel
    """
    # one example for a transformer which supports Series internally but not Panel
    cls = BoxCoxTransformer
    est = cls.create_test_instance()
    est.set_config(**backend.copy())
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert "Panel" not in inner_X_scitypes(est)
    assert "Hierarchical" not in inner_X_scitypes(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformPanelUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Panel", return_metadata=True)
    assert valid, "fit.transform does not return a Panel when given a Panel"
    # todo: possibly, add mtype check, use metadata return


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_series_in_primitives_out_supported_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Primitives"
        "fit_is_empty" = True
        "X_inner_mtype" supports "Series"

    X input to fit/transform has Series scitype
    X output from fit/transform should be Table
    """
    # one example for a transformer which supports Series internally
    cls = SummaryTransformer
    est = cls.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Primitives"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Table", return_metadata=True)
    assert valid, "fit.transform does not return a Table when given a Series"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be one, for a single series passed
    assert len(Xt) == 1


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_panel_in_primitives_out_not_supported_fit_in_transform(backend):
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Primitives"
        "fit_is_empty" = True
        "X_inner_mtype" does not support "Panel", but does supports "Series"

    X input to fit/transform has Panel scitype
    X output from fit/transform should be Table
    """
    # one example for a transformer which supports Series internally but not Panel
    cls = SummaryTransformer
    est = cls.create_test_instance()
    est.set_config(**backend.copy())
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert "Panel" not in inner_X_scitypes(est)
    assert "Hierarchical" not in inner_X_scitypes(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Primitives"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformPanelUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Table", return_metadata=True)
    assert valid, "fit.transform does not return a Table when given a Panel"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be seven = number of samples in the scenario
    assert len(Xt) == 7


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations")
    or not _check_estimator_deps(TSFreshFeatureExtractor, severity="none"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_series_in_primitives_out_not_supported_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Primitives"
        "fit_is_empty" = True
        "X_inner_mtype" supports "Panel" but does not support "Series"

    X input to fit/transform has Series scitype
    X output from fit/transform should be Table
    """
    # one example for a transformer which supports Panel internally but not Series
    cls = TSFreshFeatureExtractor
    est = cls.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Panel" in inner_X_scitypes(est)
    assert "Series" not in inner_X_scitypes(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Primitives"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Table", return_metadata=True)
    assert valid, "fit.transform does not return a Table when given a Series"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be one, for a single series passed
    assert len(Xt) == 1


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations")
    or not _check_estimator_deps(TSFreshFeatureExtractor, severity="none"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_panel_in_primitives_out_supported_with_y_in_fit_but_not_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Primitives"
        "fit_is_empty" = False
        "requires_y" = True
        "X_inner_mtype" supports "Panel"

    X input to fit/transform has Panel scitype
    X output from fit/transform should be Table
    """
    # one example for a transformer which supports Panel internally
    cls = TSFreshRelevantFeatureExtractor
    est = cls.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Panel" in inner_X_scitypes(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("requires_y")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Primitives"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformPanelUnivariateWithClassYOnlyFit()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Table", return_metadata=True)
    assert valid, "fit.transform does not return a Table when given a Panel"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be seven = number of samples in the scenario
    assert len(Xt) == 7


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_hierarchical_in_hierarchical_out_not_supported_but_series(backend):
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = False
        "X_inner_mtype" supports "Series" but not "Panel" and not "Hierarchical"

    X input to fit/transform has Hierarchical scitype
    X output from fit/transform should be Hierarchical
    """
    # one example for a transformer which supports Series internally
    cls = BoxCoxTransformer
    est = cls.create_test_instance()
    est.set_config(**backend.copy())
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert "Panel" not in inner_X_scitypes(est)
    assert "Hierarchical" not in inner_X_scitypes(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformHierarchicalUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Hierarchical", return_metadata=True)
    assert valid, "fit.transform does not return a Hierarchical when given Hierarchical"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be number of hierarchy levels times number of time points
    expected_length = len(scenario.args["transform"]["X"])
    assert len(Xt) == expected_length


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_hierarchical_in_hierarchical_out_not_supported_but_series_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = True
        "X_inner_mtype" supports "Series" but not "Panel" and not "Hierarchical"

    X input to fit/transform has Hierarchical scitype
    X output from fit/transform should be Hierarchical
    """
    # one example for a transformer which supports Series internally
    cls = ExponentTransformer
    est = cls.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert "Panel" not in inner_X_scitypes(est)
    assert "Hierarchical" not in inner_X_scitypes(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformHierarchicalUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Hierarchical", return_metadata=True)
    assert valid, "fit.transform does not return a Hierarchical when given Hierarchical"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be number of hierarchy levels times number of time points
    expected_length = len(scenario.args["transform"]["X"])
    assert len(Xt) == expected_length


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_vectorization_multivariate_no_row_vectorization(backend):
    """Test that multivariate vectorization of univariate transformers works.

    This test should trigger column (variable) vectorization, but not row vectorization.

    Setting: transformer has tags
        "univariate-only" = True
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = False
        "X_inner_mtype" supports "Series"

    X input to fit/transform has Series scitype, is multivariate
    X output from fit/transform should be Series and multivariate
    """
    # one example for a transformer which supports Series internally
    cls = BoxCoxTransformer
    est = cls.create_test_instance()
    est.set_config(**backend.copy())
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing multivariate functionality)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"
    assert est.get_tag("univariate-only")

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformSeriesMultivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Series", return_metadata=True)
    assert valid, "fit.transform does not return a Series when given a Series"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == len(scenario.args["fit"]["X"])
    assert len(Xt.columns) == len(scenario.args["fit"]["X"].columns)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_vectorization_multivariate_and_hierarchical(backend):
    """Test that fit/transform runs and returns the correct output type.

    This test should trigger both column (variable) and row (hierarchy) vectorization.

    Setting: transformer has tags
        "univariate-only" = True
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = False
        "X_inner_mtype" supports "Series" but not "Panel" and not "Hierarchical

    X input to fit/transform has Hierarchical scitype
    X output from fit/transform should be Hierarchical
    """
    # one example for a transformer which supports Series internally
    cls = BoxCoxTransformer
    est = cls.create_test_instance()
    est.set_config(**backend.copy())
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert "Panel" not in inner_X_scitypes(est)
    assert "Hierarchical" not in inner_X_scitypes(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"
    assert est.get_tag("univariate-only")

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformHierarchicalMultivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Hierarchical", return_metadata=True)
    assert valid, "fit.transform does not return a Hierarchical when given Hierarchical"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == len(scenario.args["fit"]["X"])
    assert len(Xt.columns) == len(scenario.args["fit"]["X"].columns)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_vectorization_multivariate_no_row_vectorization_empty_fit(backend):
    """Test that multivariate vectorization of univariate transformers works.

    This test should trigger column (variable) vectorization, but not row vectorization.

    Setting: transformer has tags
        "univariate-only" = True
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = True
        "X_inner_mtype" supports "Series"

    X input to fit/transform has Series scitype, is multivariate
    X output from fit/transform should be Series and multivariate
    """
    # one example for a transformer which supports Series internally
    cls = BoxCoxTransformer
    est = FitInTransform(cls.create_test_instance())
    est.set_config(**backend.copy())
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing multivariate functionality)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"
    assert est.get_tag("univariate-only")

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformSeriesMultivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Series", return_metadata=True)
    assert valid, "fit.transform does not return a Series when given a Series"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == len(scenario.args["fit"]["X"])
    assert len(Xt.columns) == len(scenario.args["fit"]["X"].columns)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_vectorization_multivariate_and_hierarchical_empty_fit(backend):
    """Test that fit/transform runs and returns the correct output type.

    This test should trigger both column (variable) and row (hierarchy) vectorization.

    Setting: transformer has tags
        "univariate-only" = True
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = True
        "X_inner_mtype" supports "Series" but not "Panel" and not "Hierarchical

    X input to fit/transform has Hierarchical scitype
    X output from fit/transform should be Hierarchical
    """
    # one example for a transformer which supports Series internally
    cls = BoxCoxTransformer
    est = FitInTransform(cls.create_test_instance())
    est.set_config(**backend.copy())
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert "Panel" not in inner_X_scitypes(est)
    assert "Hierarchical" not in inner_X_scitypes(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Series"
    assert est.get_tag("univariate-only")

    # scenario in which series are passed to fit/transform
    scenario = TransformerFitTransformHierarchicalMultivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    valid, _, _ = check_is_scitype(Xt, scitype="Hierarchical", return_metadata=True)
    assert valid, "fit.transform does not return a Hierarchical when given Hierarchical"
    # todo: possibly, add mtype check, use metadata return
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == len(scenario.args["fit"]["X"])
    assert len(Xt.columns) == len(scenario.args["fit"]["X"].columns)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_vectorize_reconstruct_unique_columns():
    """Tests that vectorization on multivariate output yields unique columns.

    Also test that the column names are as expected:
    <variable>__<transformed> if multiple transformed variables per variable are present
    <variable> if one variable is transformed to one output

    Raises
    ------
    AssertionError if output columns are not as expected.
    """
    from sktime.transformations.series.detrend import Detrender
    from sktime.transformations.series.theta import ThetaLinesTransformer

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    X_mi = get_examples("pd_multiindex_hier")[0]

    t = ThetaLinesTransformer()

    X_t_cols = t.fit_transform(X).columns

    assert set(X_t_cols) == {"a__0", "a__2", "b__0", "b__2", "c__0", "c__2"}

    X_mi_cols = t.fit_transform(X_mi)
    assert set(X_mi_cols) == {"var_0__0", "var_0__2", "var_1__0", "var_1__2"}

    X = _make_series(n_columns=2, n_timepoints=15)
    t = Detrender.create_test_instance()
    Xt = t.fit_transform(X)
    assert set(Xt.columns) == {0, 1}


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_vectorize_reconstruct_correct_hierarchy():
    """Tests correct transform return index in hierarchical case for primitives output.

    Tests that the row index is as expected if rows are vectorized over,
    by a transform that returns Primitives.
    The row index of transform return should be identical to the input,
    with temporal index level removed

    Raises
    ------
    AssertionError if output index is not as expected.
    """
    from sktime.transformations.series.summarize import SummaryTransformer
    from sktime.utils._testing.hierarchical import _make_hierarchical

    # hierarchical data with 2 variables and 2 levels
    X = _make_hierarchical(n_columns=2)

    summary_trafo = SummaryTransformer()

    # this produces a pandas DataFrame with more rows and columns
    # rows should correspond to different instances in X
    Xt = summary_trafo.fit_transform(X)

    # check that Xt.index is the same as X.index with time level dropped and made unique
    assert (X.index.droplevel(-1).unique() == Xt.index).all()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_wrong_y_is_not_passed_to_transformer():
    """Tests that y incompatible with internal type is not passed to transformer.

    Failure case of bug #6417.
    """
    from datetime import datetime

    import numpy as np
    import pandas as pd

    from sktime.pipeline import make_pipeline
    from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
    from sktime.transformations.compose import FitInTransform
    from sktime.transformations.panel.interpolate import TSInterpolator
    from sktime.transformations.series.kalman_filter import KalmanFilterTransformerFP

    # this test requires the KalmanFilterTransformerFP to be runnable
    if not _check_estimator_deps(KalmanFilterTransformerFP, severity="none"):
        return None

    # Define the multi-index
    index = pd.MultiIndex.from_tuples(
        [
            (
                0,
                datetime.strptime("2024-04-20 18:22:14.877500", "%Y-%m-%d %H:%M:%S.%f"),
            ),
            (
                0,
                datetime.strptime("2024-04-20 18:22:14.903000", "%Y-%m-%d %H:%M:%S.%f"),
            ),
            (
                1,
                datetime.strptime("2024-04-20 18:24:42.453400", "%Y-%m-%d %H:%M:%S.%f"),
            ),
            (
                1,
                datetime.strptime("2024-04-20 18:24:42.478800", "%Y-%m-%d %H:%M:%S.%f"),
            ),
        ],
        names=["instance", "Time"],
    )

    X = pd.DataFrame(
        {"LeftControllerVelocity_0": [-0.01, -0.01, 0.06, 0.06]}, index=index
    )
    y = np.array([1, 0.5])

    # noise filter only, this is a reduced MRE
    noise_filter_only = FitInTransform(KalmanFilterTransformerFP(1, denoising=True))
    noise_filter_only.fit(X, y)

    # in pipeline, this is the full MRE from bug #6417
    noise_filter = FitInTransform(KalmanFilterTransformerFP(1, denoising=True))
    interpolator = TSInterpolator(4000)
    regressor = KNeighborsTimeSeriesRegressor()

    model = make_pipeline(noise_filter, interpolator, regressor)
    model.fit(X, y)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_series_to_primitives_hierarchical():
    """Test that the correct index is returned for hierarchical Series-to-Primitives.

    Tests the vectorization case and the plain case.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Panel"

        Case 1: "X_inner_mtype" does not support Hierarchical, vectorizes
        Case 2: "X_inner_mtype" does support Hierarchical, does not vectorize

    X input to fit/transform has Hierarchical mtype
    X output from fit/transform should be Table with hierarchical index, one level less
    """
    # case 1
    # example of Series-to-Primitives not supporting Hierarchical
    cls_vectorizes = SummaryTransformer
    est = cls_vectorizes.create_test_instance()
    # ensure cls is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of cls, but we need to choose another example)
    assert "Series" in inner_X_scitypes(est)
    assert "Panel" not in inner_X_scitypes(est)
    assert "Hierarchical" not in inner_X_scitypes(est)
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Primitives"

    X = _make_hierarchical()
    Xt = est.fit_transform(X)
    ix = Xt.index

    # check that Xt.index is the same as X.index with time level dropped and made unique
    assert (X.index.droplevel(-1).unique() == ix).all()

    # case 2
    from sktime.clustering.dbscan import TimeSeriesDBSCAN
    from sktime.registry import coerce_scitype

    # example of Series-to-Primitives supporting Hierarchical
    clust = TimeSeriesDBSCAN.create_test_instance()
    est = coerce_scitype(clust, "transformer")

    # ensure est is a good example, if this fails, choose another example
    #   (if this changes, it may be due to implementing more scitypes)
    #   (then this is not a failure of est, but we need to choose another example)
    assert "Hierarchical" in inner_X_scitypes(est)
    assert est.get_tag("scitype:transform-input") == "Series"
    assert est.get_tag("scitype:transform-output") == "Primitives"

    X = _make_hierarchical()
    Xt = est.fit_transform(X)
    ix = Xt.index

    # check that Xt.index is the same as X.index with time level dropped and made unique
    assert (X.index.droplevel(-1).unique() == ix).all()
