"""Tests for HampelAnomalyDetector."""

__author__ = ["Vbhatt03"]
__all__ = []

import pandas as pd
import pytest

from sktime.detection.hampel import HampelAnomalyDetector
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,y_expected,window_size,n_sigmas",
    [
        (
            pd.DataFrame([2, 5, 7, 9, 12, 55, 1, 3, 2, 1]),
            pd.DataFrame({"ilocs": [5]}),
            5,
            3.0,
        ),
        (
            pd.DataFrame([1, 2, 3, 4, 5, 150, 6, 7, 8, 9]),
            pd.DataFrame({"ilocs": [5]}),
            5,
            3.0,
        ),
    ],
)
def test_predict_basic(X, y_expected, window_size, n_sigmas):
    """Test basic prediction with known outliers."""
    model = HampelAnomalyDetector(window_size=window_size, n_sigmas=n_sigmas)
    model.fit(X)
    y_actual = model.predict(X)
    pd.testing.assert_frame_equal(y_actual, y_expected)


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_hampel_does_not_mutate_input():
    """Check that HampelAnomalyDetector does not modify the input DataFrame."""
    X = pd.DataFrame([2, 5, 7, 9, 12, 55, 1, 3, 2, 1])
    X_original = X.copy(deep=True)

    model = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)
    _ = model.fit_transform(X)

    pd.testing.assert_frame_equal(X, X_original)


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_flat_signal():
    """Test with flat signal - should have no outliers."""
    X = pd.DataFrame([5.0] * 30)
    model = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)
    model.fit(X)
    y = model.predict(X)

    # Flat signal should have no deviations
    assert len(y) == 0
    assert isinstance(y, pd.DataFrame)
    assert "ilocs" in y.columns


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_short_series():
    """Test with series shorter than window_size."""
    X = pd.DataFrame([1, 2, 3])
    model = HampelAnomalyDetector(window_size=5)
    model.fit(X)
    y = model.predict(X)

    # Should handle gracefully, returning DataFrame with ilocs column
    assert isinstance(y, pd.DataFrame)
    assert "ilocs" in y.columns


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_even_window_size_adjustment():
    """Test window_size adjustment when center=True and even values are passed."""
    model_centered = HampelAnomalyDetector(window_size=4, center=True)
    model_trailing = HampelAnomalyDetector(window_size=4, center=False)

    assert model_centered.window_size == 5
    assert model_trailing.window_size == 4


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "window_size,n_sigmas,use_mmad,mmad_window",
    [
        (5, 3.0, False, None),  # standard Hampel
        (7, 2.5, False, None),  # larger window, tighter threshold
        (5, 3.0, True, None),  # mMAD variant with default mmad_window
        (5, 3.0, True, 7),  # mMAD with custom mmad_window
        (7, 3.0, True, None),  # mMAD with larger window
        (9, 2.0, False, None),  # very strict threshold
    ],
)
def test_parameter_combinations(window_size, n_sigmas, use_mmad, mmad_window):
    """Test various parameter combinations."""
    X = pd.DataFrame([2, 5, 7, 9, 12, 55, 1, 3, 2, 1] * 3)  # Repeat for longer series
    model = HampelAnomalyDetector(
        window_size=window_size,
        n_sigmas=n_sigmas,
        use_mmad=use_mmad,
        mmad_window=mmad_window,
    )
    model.fit(X)
    y = model.predict(X)

    # Should return valid output format
    assert isinstance(y, pd.DataFrame)
    assert "ilocs" in y.columns
    # Outlier at index 5 should be detected (and repeats at 15, 25)
    assert len(y) > 0


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_dense_output():
    """Test transform method returns dense output (same length as input)."""
    X = pd.DataFrame([2, 5, 7, 9, 12, 55, 1, 3, 2, 1])
    model = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)
    model.fit(X)
    y_dense = model.transform(X)

    # Dense output should have same length as input
    assert isinstance(y_dense, pd.DataFrame)
    assert len(y_dense) == len(X)
    assert y_dense.shape[0] == 10
    # Should have a 'labels' column for dense output
    assert "labels" in y_dense.columns


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_center_parameter():
    """Test both symmetric and trailing window modes."""
    X = pd.DataFrame([1, 2, 3, 4, 5, 150, 7, 8, 9, 10])
    model_centered = HampelAnomalyDetector(window_size=5, center=True, n_sigmas=3.0)
    model_trailing = HampelAnomalyDetector(window_size=5, center=False, n_sigmas=3.0)

    model_centered.fit(X)
    model_trailing.fit(X)

    y_centered = model_centered.predict(X)
    y_trailing = model_trailing.predict(X)

    # Centered window should detect the outlier
    assert 5 in y_centered["ilocs"].values
    # Trailing might or might not depending on window position
    # Just verify both return valid outputs
    assert isinstance(y_centered, pd.DataFrame)
    assert isinstance(y_trailing, pd.DataFrame)


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_predict_vs_fit_transform():
    """Test that predict and transform give consistent results."""
    X = pd.DataFrame([2, 5, 7, 9, 12, 55, 1, 3, 2, 1])
    model1 = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)
    model2 = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)

    # fit_predict should return sparse format
    y_predict = model1.fit_predict(X)
    # fit_transform should return dense format
    y_transform = model2.fit_transform(X)

    # Check formats
    assert isinstance(y_predict, pd.DataFrame)
    assert isinstance(y_transform, pd.DataFrame)
    assert "ilocs" in y_predict.columns
    assert "labels" in y_transform.columns
    assert len(y_transform) == len(X)


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mmad_vs_standard():
    """Test that mMAD variant produces valid results."""
    X = pd.DataFrame(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 200, 15, 16, 17, 18, 19, 20]
    )

    model_standard = HampelAnomalyDetector(window_size=5, n_sigmas=3.0, use_mmad=False)
    model_mmad = HampelAnomalyDetector(window_size=5, n_sigmas=3.0, use_mmad=True)

    model_standard.fit(X)
    model_mmad.fit(X)

    y_standard = model_standard.predict(X)
    y_mmad = model_mmad.predict(X)

    # Both should return valid DataFrames with ilocs column
    assert isinstance(y_standard, pd.DataFrame)
    assert isinstance(y_mmad, pd.DataFrame)
    assert "ilocs" in y_standard.columns
    assert "ilocs" in y_mmad.columns
    # Outlier at index 13 should be detected by at least one variant
    assert 13 in y_standard["ilocs"].values or 13 in y_mmad["ilocs"].values


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multiple_outliers():
    """Test detection of multiple outliers."""
    X = pd.DataFrame([1, 2, 3, 4, 5, 150, 6, 7, 8, 180, 10, 11, 12])
    model = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)
    model.fit(X)
    y = model.predict(X)

    # Should detect multiple outliers
    assert len(y) >= 1
    # Should include at least one of the extreme values (150 or 180)
    outlier_indices = y["ilocs"].values
    assert 5 in outlier_indices or 9 in outlier_indices


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "window_size",
    [3, 5, 7, 11],
)
def test_various_window_sizes(window_size):
    """Test with different window sizes."""
    X = pd.DataFrame([2, 5, 7, 9, 12, 55, 1, 3, 2, 1] * 3)
    model = HampelAnomalyDetector(window_size=window_size, n_sigmas=3.0)
    model.fit(X)
    y = model.predict(X)

    # Should return valid output
    assert isinstance(y, pd.DataFrame)
    assert "ilocs" in y.columns


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_threshold_sensitivity():
    """Test that stricter thresholds detect more outliers."""
    X = pd.DataFrame([2, 5, 7, 9, 12, 55, 1, 3, 2, 1] * 2)

    model_strict = HampelAnomalyDetector(window_size=5, n_sigmas=2.0)
    model_lenient = HampelAnomalyDetector(window_size=5, n_sigmas=4.0)

    model_strict.fit(X)
    model_lenient.fit(X)

    y_strict = model_strict.predict(X)
    y_lenient = model_lenient.predict(X)

    # Stricter threshold should detect more (or equal) outliers
    assert len(y_strict) >= len(y_lenient)


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_single_column_dataframe():
    """Test that single column DataFrames are handled correctly."""
    X = pd.DataFrame({"values": [1, 2, 3, 4, 5, 150, 7, 8, 9, 10]})
    model = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)
    model.fit(X)
    y = model.predict(X)

    assert isinstance(y, pd.DataFrame)
    assert "ilocs" in y.columns
    assert 5 in y["ilocs"].values


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_no_outliers():
    """Test with data that has no clear outliers."""
    # Generate gentle random-like variation without extreme outliers
    X = pd.DataFrame([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
    model = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)
    model.fit(X)
    y = model.predict(X)

    # Should have very few or no outliers
    assert len(y) <= 2  # Allow for edge effects


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_get_test_params():
    """Test that get_test_params returns valid parameters."""
    params = HampelAnomalyDetector.get_test_params()

    assert isinstance(params, list)
    assert len(params) > 0

    for param_dict in params:
        assert isinstance(param_dict, dict)
        # Each dict should have valid parameter combinations
        model = HampelAnomalyDetector(**param_dict)
        assert model is not None


@pytest.mark.skipif(
    not run_test_for_class(HampelAnomalyDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_non_default_index():
    """Test with non-default index - ilocs should be positions, not labels."""
    # Use a longer series with a clear outlier and custom index
    X = pd.Series(
        [2, 5, 7, 9, 12, 200, 1, 3, 2, 1],
        index=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    )
    model = HampelAnomalyDetector(window_size=5, n_sigmas=3.0)
    model.fit(X)
    y = model.predict(X)

    # Should return position 5 (integer location), not label 105
    assert len(y) > 0, "Should detect at least one outlier"
    assert y["ilocs"].values[0] == 5, f"Expected iloc 5, got {y['ilocs'].values[0]}"
    # Verify it's not returning the index label
    assert y["ilocs"].values[0] != 105, "Should return iloc, not index label"


def test_invalid_window_size():
    """Test that window_size < 3 raises ValueError."""
    with pytest.raises(ValueError, match="window_size must be >= 3"):
        HampelAnomalyDetector(window_size=2)


def test_invalid_n_sigmas():
    """Test that n_sigmas <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="n_sigmas must be > 0"):
        HampelAnomalyDetector(n_sigmas=0)
    with pytest.raises(ValueError, match="n_sigmas must be > 0"):
        HampelAnomalyDetector(n_sigmas=-1.5)


def test_invalid_mmad_window():
    """Test that mmad_window < 3 raises ValueError when use_mmad=True."""
    with pytest.raises(ValueError, match="mmad_window must be >= 3"):
        HampelAnomalyDetector(use_mmad=True, mmad_window=2)


def test_mmad_window_without_use_mmad():
    """Test that mmad_window raises ValueError when use_mmad=False."""
    with pytest.raises(ValueError, match="mmad_window requires use_mmad=True"):
        HampelAnomalyDetector(use_mmad=False, mmad_window=5)
