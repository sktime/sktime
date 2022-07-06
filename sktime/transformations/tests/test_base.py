# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for base class conversion and vectorization functionality.

Each test covers a "decision path" in the base class boilerplate,
    with a focus on frequently breaking paths in base class refactor and bugfixing.
The path taken depends on tags of a given transformer, and input data type.
Concrete transformer classes from sktime are imported to cover
    different combinations of transformer tags.
Transformer scenarios cover different combinations of input data types.
"""

__author__ = ["fkiraly"]
__all__ = []

from inspect import isclass

from sktime.datatypes import check_is_scitype, mtype_to_scitype
from sktime.transformations.compose import FitInTransform
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.transformations.panel.tsfresh import (
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor,
)
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.summarize import SummaryTransformer
from sktime.utils._testing.scenarios_transformers import (
    TransformerFitTransformHierarchicalMultivariate,
    TransformerFitTransformHierarchicalUnivariate,
    TransformerFitTransformPanelUnivariate,
    TransformerFitTransformPanelUnivariateWithClassYOnlyFit,
    TransformerFitTransformSeriesMultivariate,
    TransformerFitTransformSeriesUnivariate,
)

# other scenarios that might be needed later in development:
# TransformerFitTransformPanelUnivariateWithClassY,


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


def test_series_in_series_out_supported():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "scitype:transform-input" = "Series"
        "scitype:transform-output" = "Series"
        "fit_is_empty" = False
        "X_inner_mtype" supports "Series

    X input to fit/transform has Series scitype
    X ouput from fit/transform should be Series
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


def test_panel_in_panel_out_not_supported_but_series():
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


def test_panel_in_primitives_out_not_supported_fit_in_transform():
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


def test_hierarchical_in_hierarchical_out_not_supported_but_series():
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
    assert len(Xt) == 2 * 4 * 12


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
    assert len(Xt) == 2 * 4 * 12


def test_vectorization_multivariate_no_row_vectorization():
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


def test_vectorization_multivariate_and_hierarchical():
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


def test_vectorization_multivariate_no_row_vectorization_empty_fit():
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


def test_vectorization_multivariate_and_hierarchical_empty_fit():
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
