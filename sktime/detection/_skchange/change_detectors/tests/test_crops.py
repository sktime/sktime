import re

import numpy as np
import pandas as pd
import pytest

from sktime.detection._skchange.change_detectors._crops import CROPS, evaluate_segmentation
from sktime.detection._skchange.costs import GaussianCost, L1Cost, L2Cost
from sktime.detection._skchange.costs.base import BaseCost
from sktime.detection._skchange.datasets import generate_alternating_data


def test_pelt_crops():
    """Test the CROPS algorithm for path solutions to penalized CPD.

    Reference: https://arxiv.org/pdf/1412.3617
    """
    cost = GaussianCost()
    min_penalty = 0.01
    max_penalty = 10.0
    min_segment_length = 10

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=2.0,
        random_state=241,
    )

    # Fit CROPS change point detector:
    change_point_detector = CROPS(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
        min_segment_length=min_segment_length,
    )
    change_point_detector.fit(dataset)
    pruning_change_points = change_point_detector.predict(dataset.values)

    no_pruning_change_detector = CROPS(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
        min_segment_length=min_segment_length,
        prune=False,
    )
    no_pruning_change_detector.fit(dataset)
    no_pruning_changepoints = no_pruning_change_detector.predict(dataset.values)

    assert np.all(
        pruning_change_points == no_pruning_changepoints
    ), f"Expected {no_pruning_changepoints}, got {pruning_change_points}"
    # Check that the results are as expected:
    assert len(pruning_change_points) == 1

    assert no_pruning_change_detector.change_points_metadata.equals(
        change_point_detector.change_points_metadata
    ), "Expected change points metadata to be the same for both detectors."


def test_pelt_crops_with_elbow_segmentation_selection():
    """Test the CROPS algorithm for path solutions to penalized CPD.

    Reference: https://arxiv.org/pdf/1412.3617
    """
    cost = GaussianCost()
    min_penalty = 1.0e0
    max_penalty = 1.0e3
    step_size = 10

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=2.0,
        random_state=123,
    )

    # Fit CROPS change point detector:
    change_point_detector = CROPS(
        cost=cost,
        selection_method="elbow",
        step_size=step_size,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )
    change_point_detector.fit(dataset)
    pruning_change_points = change_point_detector.predict(dataset.values)

    no_pruning_change_detector = CROPS(
        cost=cost,
        selection_method="elbow",
        step_size=step_size,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
        prune=False,
    )
    no_pruning_change_detector.fit(dataset)
    no_pruning_changepoints = no_pruning_change_detector.predict(dataset.values)

    assert np.all(pruning_change_points == no_pruning_changepoints), (
        # random_state=42,
        f"Expected {no_pruning_changepoints}, got {pruning_change_points}"
    )
    # Check that the results are as expected:
    assert len(pruning_change_points) == 1

    assert no_pruning_change_detector.change_points_metadata.equals(
        change_point_detector.change_points_metadata
    ), "Expected change points metadata to be the same for both detectors."


def test_pelt_crops_raises_on_wrong_segmentation_selection():
    """Test CROPS algorithm raises error when segmentation selection is wrong."""
    cost = L2Cost()
    min_penalty = 1.0
    max_penalty = 2.0

    # Check that the results are as expected:
    with pytest.raises(ValueError):
        CROPS(
            cost=cost,
            selection_method="wrong",
            min_penalty=min_penalty,
            max_penalty=max_penalty,
        )


def test_retrieve_change_points():
    """Test the retrieve_change_points method."""
    cost = L2Cost()
    min_penalty = 40.0
    max_penalty = 50.0

    change_point_detector = CROPS(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    change_point_detector.fit(dataset)
    change_point_detector._run_crops(dataset)

    # Check that the results are as expected:
    assert len(change_point_detector.change_points_lookup[1]) == 1


def test_retrieve_change_points_2():
    """Test the retrieve_change_points method."""
    cost = L2Cost()
    min_penalty = 10.0
    max_penalty = 50.0

    change_point_detector = CROPS(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=3,
        segment_length=88,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    change_point_detector.fit(dataset)
    change_point_detector.predict(dataset)

    specific_change_points = change_point_detector.change_points_lookup[2]

    # Check that the results are as expected:
    assert np.array_equal(
        specific_change_points, np.array([88, 176])
    ), f"Expected [88, 176], got {specific_change_points}"


def test_non_aggregated_cost_raises():
    """Test CROPS algorithm raises an error if a non-aggregated cost is used."""
    cost = L1Cost()
    two_dim_data = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=2,  # Two dimensions
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    crops_cpd = CROPS(cost=cost, min_penalty=1.0, max_penalty=2.0)
    crops_cpd.fit(two_dim_data)

    with pytest.raises(
        ValueError,
        match="CROPS only supports costs that return a single value per cut",
    ):
        crops_cpd.predict(two_dim_data.values)


def test_crops_with_min_segment_length_greater_than_step_size_raises():
    """Test CROPS raises an error if min_segment_length is greater than step_size."""
    cost = L2Cost()
    min_penalty = 1.0
    max_penalty = 2.0
    min_segment_length = 10

    with pytest.raises(
        ValueError,
        match=re.escape(
            "CROPS `min_segment_length`(=10) cannot "
            "be greater than the `step_size`(=5) > 1."
        ),
    ):
        CROPS(
            cost=cost,
            min_penalty=min_penalty,
            max_penalty=max_penalty,
            min_segment_length=min_segment_length,
            step_size=5,  # Step size is less than min_segment_length
        )


@pytest.mark.parametrize("CostClass", [L2Cost, L1Cost, GaussianCost])
def test_evaluate_segmentation(CostClass: type[BaseCost]):
    cost = CostClass.create_test_instance()
    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=1, random_state=5)
    cost.fit(df)
    np_segmentation = np.array([0, 10, 20, 30, 40, 50])
    pd_segmentation = pd.Series(np_segmentation)
    np_changepoints = np.array([10, 20, 30, 40])

    np_2d_segmentation = np_segmentation.reshape(-1, 1)

    assert np.array_equal(
        evaluate_segmentation(cost, np_segmentation),
        evaluate_segmentation(cost, pd_segmentation),
    )
    assert np.array_equal(
        evaluate_segmentation(cost, np_2d_segmentation),
        evaluate_segmentation(cost, np_segmentation),
    )
    assert np.array_equal(
        evaluate_segmentation(cost, np_changepoints),
        evaluate_segmentation(cost, pd_segmentation),
    )


@pytest.mark.parametrize("CostClass", [L2Cost, L1Cost, GaussianCost])
def test_evaluate_segmentation_raises(CostClass: type[BaseCost]):
    cost = CostClass.create_test_instance()
    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=1, random_state=5)
    cost.fit(df)

    with pytest.raises(
        ValueError,
        match="The segmentation must contain strictly increasing entries.",
    ):
        # Not strictly increasing segmentation:
        evaluate_segmentation(cost, np.array([0, 10, 20, 30, 20, 40]))

    with pytest.raises(ValueError, match="The segmentation must univariate"):
        # Invalid segmentation shape:
        evaluate_segmentation(cost, np.array([[0, 10], [20, 30], [40, 50]]))
