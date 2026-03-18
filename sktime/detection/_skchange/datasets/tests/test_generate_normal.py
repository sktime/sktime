"""Tests for the generate_piecewise_normal_data function."""

import numpy as np
import pytest

from sktime.detection._skchange.datasets import (
    generate_anomalous_data,
    generate_changing_data,
    generate_piecewise_normal_data,
)


def test_generate_piecewise_normal_data_default_params():
    """Test that the function generates data with default parameters."""
    df = generate_piecewise_normal_data()
    assert not df.empty


@pytest.mark.parametrize(
    "means",
    [0, 1, np.array([0, 1]), [0, 1], [np.array([0, 5]), np.array([0, 0])], None],
)
def test_generate_piecewise_normal_data_valid_means(
    means: float | np.ndarray | list[float] | list[np.ndarray] | None,
):
    """Test that the function generates data with the correct means."""
    n_segments = 2
    df, params = generate_piecewise_normal_data(
        means=means,
        n_samples=10,
        n_variables=2,
        n_segments=n_segments,
        return_params=True,
    )
    expected_n_segments = (
        n_segments if not isinstance(means, list) or means is None else len(means)
    )
    assert len(params["means"]) == expected_n_segments


def test_generate_piecewise_normal_data_invalid_means():
    """Test that the function generates data with the correct means."""
    with pytest.raises((ValueError, IndexError)):
        generate_piecewise_normal_data(means=[], n_samples=10, n_variables=1)
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(
            means=np.array([0, 0]), n_samples=10, n_variables=3
        )


@pytest.mark.parametrize(
    "variances",
    [
        1,
        2,
        np.array([1, 2]),
        np.diag([1, 2]),
        [1, 2],
        [np.array([1, 5]), np.array([1, 1])],
        None,
    ],
)
def test_generate_piecewise_normal_data_valid_variances(
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None,
):
    """Test that the function generates data with the correct variances."""
    n_segments = 2
    df, params = generate_piecewise_normal_data(
        variances=variances,
        n_samples=10,
        n_variables=2,
        n_segments=n_segments,
        return_params=True,
    )
    expected_n_segments = (
        n_segments
        if not isinstance(variances, list) or variances is None
        else len(variances)
    )
    assert len(params["variances"]) == expected_n_segments


def test_generate_piecewise_normal_data_invalid_variances():
    """Test that the function raises ValueError for invalid variances."""
    with pytest.raises((ValueError, IndexError)):
        generate_piecewise_normal_data(variances=[], n_samples=10, n_variables=1)
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(
            n_samples=10, n_variables=3, variances=np.array([1, 1])
        )
    with pytest.raises(ValueError):
        unsymmetric_cov = np.array([[1, 2], [0, 1]])
        generate_piecewise_normal_data(
            n_samples=10, n_variables=2, variances=unsymmetric_cov
        )
    with pytest.raises(ValueError):
        singular_cov = np.array([[1, 0], [0, 0]])
        generate_piecewise_normal_data(
            n_samples=10, n_variables=2, variances=singular_cov
        )
    with pytest.raises(ValueError):
        non_square_cov = np.array([[1, 0], [0, 1], [0, 0]])
        generate_piecewise_normal_data(
            n_samples=10, n_variables=2, variances=non_square_cov
        )
    with pytest.raises(ValueError):
        cov_3d = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        generate_piecewise_normal_data(n_samples=10, n_variables=2, variances=cov_3d)


@pytest.mark.parametrize("lengths", [5, [3, 5, 7], np.repeat(1, 10), None])
def test_generate_piecewise_normal_data_valid_lengths(lengths: list[int]):
    """Test that the function generates data with the correct lengths."""
    df, params = generate_piecewise_normal_data(
        n_samples=10,
        n_variables=1,
        lengths=lengths,
        return_params=True,
    )
    assert len(params["lengths"]) > 0
    if isinstance(lengths, list):
        assert np.all(np.array(params["lengths"]) == np.asarray(lengths))


@pytest.mark.parametrize("lengths", [-1, [], [1, -1]])
def test_generate_piecewise_normal_data_invalid_lengths(lengths: list[int]):
    """Test that the function raises ValueError for invalid lengths."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(n_samples=10, n_variables=2, lengths=lengths)


@pytest.mark.parametrize("n_samples", [1, 2, 10, 100])
def test_generate_piecewise_normal_data_valid_n_samples(n_samples: int):
    """Test that the function generates data with the correct number of rows."""
    df = generate_piecewise_normal_data(n_samples=n_samples, n_segments=1, seed=43)
    assert df.shape[0] == n_samples


@pytest.mark.parametrize("n_samples", [-1, 0, 0.5])
def test_generate_piecewise_normal_data_invalid_n_samples(n_samples: int):
    """Test that the function raises ValueError for invalid n_samples values."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(n_samples=n_samples, seed=43)


@pytest.mark.parametrize("n_segments", [1, 2, 10])
def test_generate_piecewise_normal_data_n_segments(n_segments: int):
    """Test that the function generates data with correct number of segments."""
    df, params = generate_piecewise_normal_data(
        n_samples=10, n_segments=n_segments, return_params=True
    )
    lengths = params["lengths"]
    if n_segments is not None:
        assert len(lengths) == n_segments
    else:
        assert len(lengths) >= 1


@pytest.mark.parametrize("n_segments", [-1, 0])
def test_generate_piecewise_normal_data_invalid_n_segments(n_segments: int):
    """Test that the function raises ValueError for invalid n_segments values."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(n_samples=10, n_segments=n_segments)


@pytest.mark.parametrize("n_variables", [1, 2, 10, 100])
def test_generate_piecewise_normal_data_valid_n_variables(n_variables: int):
    """Test that the function generates data with the correct number of columns."""
    df = generate_piecewise_normal_data(n_samples=10, n_variables=n_variables)
    assert df.shape[1] == n_variables


@pytest.mark.parametrize("n_variables", [-1, 0, 0.5])
def test_generate_piecewise_normal_data_invalid_n_variables(n_variables: int):
    """Test that the function raises ValueError for invalid n_variables values."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(n_samples=10, n_variables=n_variables)


@pytest.mark.parametrize("randomise_affected_variables", [True, False])
def test_generate_piecewise_normal_data_valid_proportion_affected(
    randomise_affected_variables: bool,
):
    """Test that the function generates data with the correct proportion_affected."""
    n_samples = 20
    n_variables = 4
    proportion_affected = 0.5
    df, params = generate_piecewise_normal_data(
        n_samples=n_samples,
        n_variables=n_variables,
        n_segments=4,
        proportion_affected=proportion_affected,
        randomise_affected_variables=randomise_affected_variables,
        return_params=True,
    )

    expected_nonzero_changed_mean = int(np.ceil(n_variables * proportion_affected))
    for prev_mean, curr_mean in zip(params["means"][:-1], params["means"][1:]):
        assert np.count_nonzero(curr_mean - prev_mean) == expected_nonzero_changed_mean


@pytest.mark.parametrize(
    "proportion_affected",
    [
        -0.1,
        1.1,
        2,
        [],
    ],
)
def test_generate_piecewise_normal_data_invalid_proportion_affected(
    proportion_affected: float | str | None,
):
    """Test that the function raises ValueError for invalid proportion_affected."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(
            n_samples=10,
            n_variables=2,
            n_segments=2,
            proportion_affected=proportion_affected,
        )


def test_generate_changing_data():
    n = 100
    generate_changing_data(
        n=n,
        changepoints=[25, 50],
        means=[3.0, 0.0, 5.0],
        variances=[1.0, 2.0, 3.0],
    )

    generate_changing_data(
        n,
        changepoints=40,
    )

    generate_changing_data(
        n=n,
        changepoints=[],
        means=1.0,
        variances=5.0,
    )


def test_generate_changing_data_with_multiple_changepoints():
    n = 100
    changepoints = [10, 30]  # 3 segements
    means = [1.0, 3.0]  # 2 means -> 3 is required.
    variances = [1.0, 3.0]  # 2 variances -> 3 is required.
    seed = 1
    with pytest.raises(ValueError):
        generate_changing_data(
            n=n,
            changepoints=changepoints,
            means=means,
            variances=variances,
            random_state=seed,
        )


def test_generate_changing_data_invalid_changepoints():
    with pytest.raises(ValueError):
        generate_changing_data(
            n=100,
            changepoints=110,  # Invalid changepoint.
        )
    with pytest.raises(ValueError):
        generate_changing_data(
            n=100,
            changepoints=[10, 20, 20],  # Negative changepoint.
        )


def test_generate_changing_data_mismatched_lengths():
    n = 100
    changepoints = [10, 20]
    means = [1.0, 2.0, 3.0]
    variances = [1.0, 2.0]  # Mismatched length
    seed = 1
    with pytest.raises(ValueError):
        generate_changing_data(
            n=n,
            changepoints=changepoints,
            means=means,
            variances=variances,
            random_state=seed,
        )


@pytest.mark.parametrize(
    "anomalies",
    [
        (10, 20, 30),  # Need to be a 2-tuple or list of 2-tuples
        (30, 20),  # Start must be less than end
        (-3, 10),  # Start must be non-negative
        (50, 110),  # End must be less than n
    ],
)
def test_generate_anomalous_data_invalid_anomalies(anomalies: tuple | list[tuple]):
    """Test that the function raises ValueError for invalid anomalies."""
    with pytest.raises(ValueError):
        generate_anomalous_data(
            n=100,
            means=[10],
            anomalies=anomalies,
        )
    with pytest.raises(ValueError):
        generate_anomalous_data(
            n=100,
            means=[10, 20],
            variances=[1, 2],
            anomalies=[(10, 20)],
        )
