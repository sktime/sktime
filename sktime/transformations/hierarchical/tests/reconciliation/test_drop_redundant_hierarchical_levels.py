import pandas as pd
import pytest

from sktime.transformations.hierarchical.reconciliation.forecast_proportions import (
    DropRedundantHierarchicalLevels,
)


@pytest.fixture
def hierarchical_data():
    """Fixture for a sample hierarchical DataFrame with redundant levels."""
    index = pd.MultiIndex.from_tuples(
        [
            ("__total", "__total", pd.Period("2020-01-01")),
            ("stateA", "regionA", pd.Period("2020-01-01")),
            ("stateA", "regionB", pd.Period("2020-01-01")),
            ("stateA", "regionC", pd.Period("2020-01-01")),
        ],
        names=["level_0", "level_1", "time"],
    )
    data = pd.DataFrame({"value": [100, 40, 30, 30]}, index=index)
    return data


def test_fit(hierarchical_data):
    """Test the `_fit` method to ensure it detects levels to drop."""
    transformer = DropRedundantHierarchicalLevels()
    transformer.fit(hierarchical_data)

    assert hasattr(transformer, "levels_to_drop_")
    assert transformer.levels_to_drop_ == [0], "Expected to drop the first level."


def test_transform(hierarchical_data):
    """Test the `_transform` method to ensure it drops redundant levels."""
    transformer = DropRedundantHierarchicalLevels()
    transformer.fit(hierarchical_data)
    transformed = transformer.transform(hierarchical_data, None)

    assert (
        transformed.index.nlevels == 2
    ), "Expected the transformed index to have 2 levels."
    assert (
        "level_0" not in transformed.index.names
    ), "The redundant level was not removed."


def test_inverse_transform(hierarchical_data):
    """Test the `_inverse_transform`, to ensure it reconstructs the index."""
    transformer = DropRedundantHierarchicalLevels()
    transformer.fit(hierarchical_data)
    transformed = transformer.transform(hierarchical_data)
    inversed = transformer.inverse_transform(transformed)

    pd.testing.assert_frame_equal(hierarchical_data, inversed)
    assert (
        inversed.index.nlevels == hierarchical_data.index.nlevels
    ), "Expected the index to match the original."


def test_no_hierarchy_handling():
    """Test the transformer with a non-hierarchical DataFrame."""
    non_hierarchical_data = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])
    transformer = DropRedundantHierarchicalLevels()
    transformer.fit(non_hierarchical_data)

    assert transformer._no_hierarchy is True, "Expected `_no_hierarchy` to be True."
    transformed = transformer.transform(non_hierarchical_data)
    pd.testing.assert_frame_equal(non_hierarchical_data, transformed)
