# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for _deps dependency aggregation in _HeterogenousMetaEstimator.

Tests in this module:

    test_deps_single_component - tests _deps with a single component
    test_deps_multiple_components - tests _deps with multiple components
    test_deps_deduplication - tests that duplicate dependencies are deduplicated
    test_deps_nested_composite - tests _deps with nested composites
    test_deps_unfitted - tests _deps works for unfitted composites
    test_deps_no_dependencies - tests _deps with components that have no dependencies
    test_deps_preserve_own - tests that composite's own dependencies are preserved
    test_deps_external_object - tests _deps with non-BaseObject components
    test_deps_nested_list - tests nested list dependency structures
"""

__all__ = [
    "test_deps_single_component",
    "test_deps_multiple_components",
    "test_deps_deduplication",
    "test_deps_nested_composite",
    "test_deps_unfitted",
    "test_deps_no_dependencies",
    "test_deps_preserve_own",
    "test_deps_nested_list",
    "test_deps_deduplication_nested",
    "test_deps_multiple_components_with_nested_lists",
    "test_deps_nested_composite_ignores_aggregate_override",
    "test_deps_structurally_distinct_nested_lists",
    "test_deps_real_transformer_pipeline_recompute",
    "test_deps_dynamic_real_composites",
]

import pytest

from sktime.base import BaseObject
from sktime.base._meta import _HeterogenousMetaEstimator


class DummyEstimatorWithDeps(BaseObject):
    """Dummy estimator with python_dependencies tag for testing."""

    _tags = {"python_dependencies": "numpy"}


class DummyEstimatorWithMultipleDeps(BaseObject):
    """Dummy estimator with multiple python_dependencies for testing."""

    _tags = {"python_dependencies": ["numpy", "scipy"]}


class DummyEstimatorWithNestedDeps(BaseObject):
    """Dummy estimator with nested list dependencies for testing (OR semantics)."""

    _tags = {"python_dependencies": [["numpy>=1.20", "pandas>=1.3"], "scikit-learn>=0.24"]}


class DummyEstimatorNoDeps(BaseObject):
    """Dummy estimator without python_dependencies tag for testing."""

    pass


class DummyEstimatorOwnDeps(BaseObject):
    """Dummy estimator with its own dependencies for testing."""

    _tags = {"python_dependencies": "pandas"}


class DummyComposite(_HeterogenousMetaEstimator, BaseObject):
    """Dummy composite for testing _deps without real sktime imports."""

    _steps_attr = "_steps"

    def __init__(self, steps):
        self.steps = steps
        super().__init__()

    @property
    def _steps(self):
        return self.steps

    @_steps.setter
    def _steps(self, value):
        self.steps = value


def test_deps_single_component():
    """Test _deps with a single component with dependencies."""
    pipeline = DummyComposite([("estimator", DummyEstimatorWithDeps())])

    deps = pipeline._deps()
    assert isinstance(deps, list)
    assert "numpy" in deps


def test_deps_multiple_components():
    """Test _deps with multiple components with different dependencies."""
    pipeline = DummyComposite(
        [
            ("estimator1", DummyEstimatorWithDeps()),
            ("estimator2", DummyEstimatorWithMultipleDeps()),
        ]
    )

    deps = pipeline._deps()
    assert isinstance(deps, list)
    assert "numpy" in deps
    assert "scipy" in deps


def test_deps_deduplication():
    """Test that duplicate dependencies are deduplicated."""
    pipeline = DummyComposite(
        [
            ("estimator1", DummyEstimatorWithDeps()),
            ("estimator2", DummyEstimatorWithDeps()),
            ("estimator3", DummyEstimatorWithMultipleDeps()),
        ]
    )

    deps = pipeline._deps()
    assert isinstance(deps, list)
    # numpy should appear only once despite being in multiple components
    assert deps.count("numpy") == 1
    assert "scipy" in deps


def test_deps_nested_composite():
    """Test _deps with nested composite/pipeline dependencies."""
    inner_pipeline = DummyComposite([("estimator", DummyEstimatorWithDeps())])

    outer_pipeline = DummyComposite(
        [
            ("inner", inner_pipeline),
            ("estimator", DummyEstimatorWithMultipleDeps()),
        ]
    )

    deps = outer_pipeline._deps()
    assert isinstance(deps, list)
    assert "numpy" in deps
    assert "scipy" in deps


def test_deps_unfitted():
    """Test _deps works for unfitted composites."""
    pipeline = DummyComposite(
        [
            ("estimator1", DummyEstimatorWithDeps()),
            ("estimator2", DummyEstimatorWithMultipleDeps()),
        ]
    )

    # Should work without fitting
    deps = pipeline._deps()
    assert isinstance(deps, list)
    assert "numpy" in deps
    assert "scipy" in deps


def test_deps_no_dependencies():
    """Test _deps with components that have no dependencies."""
    pipeline = DummyComposite(
        [
            ("estimator1", DummyEstimatorNoDeps()),
            ("estimator2", DummyEstimatorNoDeps()),
        ]
    )

    deps = pipeline._deps()
    assert isinstance(deps, list)
    assert len(deps) == 0


def test_deps_preserve_own():
    """Test that composite's own dependencies are preserved in dynamic tag."""
    class CompositeWithOwnDeps(DummyComposite):
        _tags = {"python_dependencies": "pandas"}

    pipeline = CompositeWithOwnDeps(
        [
            ("estimator1", DummyEstimatorWithDeps()),
            ("estimator2", DummyEstimatorWithMultipleDeps()),
        ]
    )

    # Check that component dependencies are collected
    component_deps = pipeline._deps()
    assert isinstance(component_deps, list)
    assert "numpy" in component_deps
    assert "scipy" in component_deps

    # Check that composite's own dependencies are still accessible
    own_deps = pipeline.get_class_tag("python_dependencies")
    assert own_deps == "pandas"

    # Check that combining works correctly
    all_deps = ["pandas", component_deps]
    combined = pipeline._combine_dependencies(all_deps)
    assert isinstance(combined, list)
    assert "pandas" in combined
    assert "numpy" in combined
    assert "scipy" in combined


def test_deps_external_object():
    """Test _deps with non-BaseObject components."""
    class NotABaseObject:
        def get_tag(self, tag_name, tag_value_default=None):
            return tag_value_default

    pipeline = DummyComposite(
        [
            ("external", NotABaseObject()),
            ("estimator", DummyEstimatorWithDeps()),
        ]
    )

    # Should handle the non-BaseObject gracefully via get_tag returning None
    deps = pipeline._deps()
    assert isinstance(deps, list)
    assert "numpy" in deps


def test_deps_nested_list():
    """Test that nested list dependency structures are preserved."""
    pipeline = DummyComposite([("estimator", DummyEstimatorWithNestedDeps())])

    deps = pipeline._deps()
    assert deps == [
        ["numpy>=1.20", "pandas>=1.3"],
        "scikit-learn>=0.24",
    ]


def test_deps_deduplication_nested():
    """Test deduplication with nested dependency structures."""
    class MockEstimatorWithNested(BaseObject):
        _tags = {"python_dependencies": [["numpy", "scipy"], "pandas"]}

    pipeline = DummyComposite([
        ("est1", MockEstimatorWithNested()),
        ("est2", MockEstimatorWithNested()),
    ])

    deps = pipeline._deps()
    assert deps == [["numpy", "scipy"], "pandas"]


def test_deps_multiple_components_with_nested_lists():
    """Test that OR expressions from multiple components remain separate."""

    class MockEstimatorWithOtherNested(BaseObject):
        _tags = {"python_dependencies": [["darts", "u8darts"], "pandas"]}

    pipeline = DummyComposite(
        [
            ("first", DummyEstimatorWithNestedDeps()),
            ("second", MockEstimatorWithOtherNested()),
        ]
    )

    assert pipeline._deps() == [
        ["numpy>=1.20", "pandas>=1.3"],
        "scikit-learn>=0.24",
        ["darts", "u8darts"],
        "pandas",
    ]


def test_deps_nested_composite_ignores_aggregate_override():
    """Test nested composites use direct tags and recurse into components."""

    class NestedComposite(DummyComposite):
        _tags = {"python_dependencies": "nested-own"}

    nested = NestedComposite([("leaf", DummyEstimatorWithDeps())])
    nested.set_tags(**{"python_dependencies": ["stale", "aggregate"]})
    outer = DummyComposite([("nested", nested)])

    assert outer._deps() == ["nested-own", "numpy"]


def test_deps_structurally_distinct_nested_lists():
    """Test top-level AND and nested OR grouping are not conflated."""

    class FirstShape(BaseObject):
        _tags = {"python_dependencies": ["numpy", ["pandas", "polars"]]}

    class SecondShape(BaseObject):
        _tags = {"python_dependencies": [["numpy", "pandas"], "polars"]}

    first = DummyComposite([("estimator", FirstShape())])
    second = DummyComposite([("estimator", SecondShape())])

    assert first._deps() == ["numpy", ["pandas", "polars"]]
    assert second._deps() == [["numpy", "pandas"], "polars"]


def test_deps_real_transformer_pipeline_recompute():
    """Test dynamic aggregation is idempotent and clears replaced dependencies."""
    from sktime.transformations.compose import TransformerPipeline
    from sktime.transformations.exponent import ExponentTransformer

    class TransformerWithDeps(ExponentTransformer):
        _tags = {"python_dependencies": ["dep-a", ["dep-b", "dep-c"]]}

    class ReplacementTransformer(ExponentTransformer):
        _tags = {"python_dependencies": "dep-replacement"}

    class PipelineWithOwnDeps(TransformerPipeline):
        _tags = {"python_dependencies": "dep-own"}

    pipeline = TransformerPipeline([TransformerWithDeps()])
    expected = ["dep-a", ["dep-b", "dep-c"]]
    pipeline.__dynamic_tags__()
    assert pipeline.get_tag("python_dependencies") == expected
    pipeline.__dynamic_tags__()
    assert pipeline.get_tag("python_dependencies") == expected

    pipeline.steps = [ReplacementTransformer()]
    pipeline.__dynamic_tags__()
    assert pipeline.get_tag("python_dependencies") == ["dep-replacement"]

    owned_pipeline = PipelineWithOwnDeps([TransformerWithDeps()])
    assert owned_pipeline.get_tag("python_dependencies") == [
        "dep-own",
        "dep-a",
        ["dep-b", "dep-c"],
    ]


def test_deps_dynamic_real_composites():
    """Test dynamic dependency tags on the concrete pipeline composites."""
    try:
        from sktime.forecasting.compose import (
            ForecastingPipeline,
            TransformedTargetForecaster,
        )
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.exponent import ExponentTransformer
    except (ImportError, OSError) as exc:
        pytest.skip(f"real composite imports unavailable: {exc}")

    class TransformerWithDeps(ExponentTransformer):
        _tags = {"python_dependencies": ["dep-a", ["dep-b", "dep-c"]]}

    transformer = TransformerWithDeps()
    expected = ["dep-a", ["dep-b", "dep-c"]]

    forecasting_pipeline = ForecastingPipeline([transformer, NaiveForecaster()])
    target_pipeline = TransformedTargetForecaster([transformer, NaiveForecaster()])
    transformer_pipeline = TransformerPipeline([transformer, TransformerWithDeps()])

    assert forecasting_pipeline.get_tag("python_dependencies") == expected
    assert target_pipeline.get_tag("python_dependencies") == expected
    assert transformer_pipeline.get_tag("python_dependencies") == expected

    nested_pipeline = TransformerPipeline([transformer])
    outer_pipeline = TransformerPipeline([nested_pipeline, transformer])
    assert outer_pipeline.get_tag("python_dependencies") == expected