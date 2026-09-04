"""Tests for PAA utilities."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.dictionary_based._paa import PAAlegacy as PAA


@pytest.mark.skipif(
    not run_test_for_class(PAA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 11, 0])
def test_bad_input_args(bad_num_intervals):
    """Test that exception is raised for bad num intervals."""
    X = pd.DataFrame(np.ones(10))

    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            PAA(num_intervals=bad_num_intervals).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            PAA(num_intervals=bad_num_intervals).fit(X).transform(X)


@pytest.mark.skipif(
    not run_test_for_class(PAA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_output_of_transformer():
    """Test that the transformer has changed the data correctly."""
    X = pd.DataFrame(np.arange(1, 11))

    p = PAA(num_intervals=3).fit(X)
    res = p.transform(X)

    # Convert output to numpy array so the test works with
    # sktime-compatible output containers.
    actual_values = np.asarray(res).reshape(-1)

    # PAA with 10 observations and 3 intervals.
    expected_values = np.array([2.2, 5.5, 8.8])

    assert np.allclose(
        actual_values,
        expected_values,
        rtol=1e-5,
        atol=1e-8,
    )


@pytest.mark.skipif(
    not run_test_for_class(PAA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_output_dimensions():
    """Test output dimensions."""

    # Test with univariate - 12 timepoints, 1 feature
    X = pd.DataFrame(np.ones(12))

    p = PAA(num_intervals=5).fit(X)
    res = p.transform(X)

    # Convert to numpy array for dimension checks.
    res_array = np.asarray(res)

    assert res_array.shape == (5, 1)

    # Test with multivariate - 12 timepoints, 5 features
    X = pd.DataFrame(np.ones((12, 5)))

    p = PAA(num_intervals=5).fit(X)
    res = p.transform(X)

    res_array = np.asarray(res)

    assert res_array.shape == (5, 5)


@pytest.mark.skipif(
    not run_test_for_class(PAA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_paa_performs_correctly_along_each_dim():
    """Test that PAA produces the same result along each dimension."""

    # 10 timepoints, 2 features
    X = pd.DataFrame(
        np.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
                [6, 6],
                [7, 7],
                [8, 8],
                [9, 9],
                [10, 10],
            ]
        )
    )

    p = PAA(num_intervals=3).fit(X)
    res = p.transform(X)

    res_array = np.asarray(res)

    expected_values = np.array([2.2, 5.5, 8.8])

    # Check both columns have same result
    for col in range(res_array.shape[1]):
        actual_values = res_array[:, col]

        assert np.allclose(
            actual_values,
            expected_values,
            rtol=1e-5,
            atol=1e-8,
        )
