import itertools

import pandas as pd
import pytest

from sktime.transformations.hierarchical.reconcile._topdown import (
    SqueezeHierarchy,
)


@pytest.fixture
def hierarchical_data():
    """Fixture for a sample hierarchical DataFrame with redundant levels."""
    index = pd.MultiIndex.from_tuples(
        [
            ("__total", "__total", "__total", pd.Period("2020-01-01")),
            ("level0_1", "stateA", "regionA", pd.Period("2020-01-01")),
            ("level0_1", "regionB", pd.Period("2020-01-01")),
            ("stateA", "regionC", pd.Period("2020-01-01")),
        ],
        names=["level_0", "level_1", "time"],
    )
    data = pd.DataFrame({"value": [100, 40, 30, 30]}, index=index)
    return data


def create_redundant_hierarchical_indexes(
    n_hier_levels, n_redundant, n_instances_per_level
):
    assert n_redundant < n_hier_levels, (
        "Number of redundant levels must be less than the number of levels."
    )

    level_and_subvalues = [
        [f"level{l}_{i}" for i in range(n_instances_per_level)]
        for l in range(n_hier_levels)
    ]

    # Force a single value for the redundant levels
    for i in range(n_redundant):
        level_and_subvalues[i] = level_and_subvalues[i][:1]

    # Create tuples (level0_0, level1_0, level2_0, ..., levelN_0) with a product
    # of all the subvalues
    indexes = list(itertools.product(*level_and_subvalues))

    # add __total index
    indexes.append(["__total"] * n_hier_levels)
    # Add a time index
    indexes = [(*i, pd.Period("2020-01-01")) for i in indexes]

    return pd.DataFrame(
        {"value": [1] * len(indexes)},
        index=pd.MultiIndex.from_tuples(
            indexes, names=[f"level_{i}" for i in range(n_hier_levels)] + ["time"]
        ),
    ).sort_index()


@pytest.mark.parametrize(
    "n_redundant,n_hier_levels", [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
)
def test_fit(n_redundant, n_hier_levels, n_instances_per_level=2):
    """Test the `_fit` method to ensure it detects levels to drop."""

    X = create_redundant_hierarchical_indexes(
        n_hier_levels, n_redundant, n_instances_per_level
    )
    transformer = SqueezeHierarchy()
    transformer.fit(X)

    expected_levels_to_drop = min(n_redundant, n_hier_levels - 2)

    assert hasattr(transformer, "levels_to_drop_")
    assert len(transformer.levels_to_drop_) == expected_levels_to_drop, (
        "Expected to drop the first level."
    )


@pytest.mark.parametrize(
    "n_redundant,n_hier_levels", [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
)
def test_transform(n_redundant, n_hier_levels, n_instances_per_level=2):
    """Test the `_transform` method to ensure it drops redundant levels."""
    X = create_redundant_hierarchical_indexes(
        n_hier_levels, n_redundant, n_instances_per_level
    )
    transformer = SqueezeHierarchy()
    transformer.fit(X)
    transformed = transformer.transform(X, None)

    # At least 2 leves besides the timeindex, since Hierarchical representation
    # always has at least 3 levels by definition
    expected = max(n_hier_levels - n_redundant, 2)
    assert transformed.index.nlevels - 1 == expected, (
        "Expected the transformed index to have 3 levels."
    )


@pytest.mark.parametrize(
    "n_redundant,n_hier_levels", [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
)
def test_inverse_transform(n_redundant, n_hier_levels, n_instances_per_level=2):
    """Test the `_inverse_transform`, to ensure it reconstructs the index."""
    X = create_redundant_hierarchical_indexes(
        n_hier_levels, n_redundant, n_instances_per_level
    )
    transformer = SqueezeHierarchy()
    transformer.fit(X)
    transformed = transformer.transform(X)
    inversed = transformer.inverse_transform(transformed)

    pd.testing.assert_frame_equal(X, inversed)
    assert inversed.index.nlevels == X.index.nlevels, (
        "Expected the index to match the original."
    )


def test_no_hierarchy_handling():
    """Test the transformer with a non-hierarchical DataFrame."""
    non_hierarchical_data = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])
    transformer = SqueezeHierarchy()
    transformer.fit(non_hierarchical_data)

    assert transformer._no_hierarchy is True, "Expected `_no_hierarchy` to be True."
    transformed = transformer.transform(non_hierarchical_data)
    pd.testing.assert_frame_equal(non_hierarchical_data, transformed)
