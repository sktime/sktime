"""Standalone test for OverlayDX metric without full sktime dependencies."""

import numpy as np


def overlay_dx_score_simple(y_true, y_pred, max_percentage=100.0, min_percentage=0.1, step=0.1):
    """Simplified version of overlay_dx for testing."""
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Compute absolute errors
    abs_errors = np.abs(y_true_np - y_pred_np)
    
    # Get value range
    value_range = np.ptp(y_true_np)  # peak-to-peak (max - min)
    value_range = max(value_range, 1e-10)  # Avoid division by zero
    
    # Generate tolerance thresholds
    thresholds = np.arange(min_percentage, max_percentage + step, step)
    
    # Compute percentage within tolerance for each threshold
    percentages_within = []
    n_samples = len(y_true_np)
    
    for threshold_pct in thresholds:
        tolerance = (threshold_pct / 100.0) * value_range
        within_tolerance = np.sum(abs_errors <= tolerance)
        pct_within = (within_tolerance / n_samples) * 100.0
        percentages_within.append(pct_within)
    
    # Compute area under curve (AUC) using trapezoidal rule
    # Use scipy.integrate.trapezoid or manual calculation
    try:
        from scipy.integrate import trapezoid
        auc = trapezoid(percentages_within, thresholds)
    except ImportError:
        # Manual trapezoidal integration
        auc = 0.0
        for i in range(len(thresholds) - 1):
            auc += (percentages_within[i] + percentages_within[i+1]) / 2.0 * (thresholds[i+1] - thresholds[i])
    
    # Normalize by the range of thresholds
    normalized_score = auc / (max_percentage - min_percentage)
    
    return normalized_score


# Test cases
print("Testing OverlayDX metric implementation...")
print("=" * 60)

# Test 1: Perfect prediction
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
score = overlay_dx_score_simple(y_true, y_pred)
print(f"Test 1 - Perfect prediction:")
print(f"  y_true: {y_true}")
print(f"  y_pred: {y_pred}")
print(f"  Score: {score:.2f}")
print(f"  Expected: High score (>90)")
print(f"  Result: {'PASS' if score > 90 else 'FAIL'}")
print()

# Test 2: Good prediction
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
score = overlay_dx_score_simple(y_true, y_pred)
print(f"Test 2 - Good prediction (small errors):")
print(f"  y_true: {y_true}")
print(f"  y_pred: {y_pred}")
print(f"  Score: {score:.2f}")
print(f"  Expected: High score (>80)")
print(f"  Result: {'PASS' if score > 80 else 'FAIL'}")
print()

# Test 3: Poor prediction
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
score = overlay_dx_score_simple(y_true, y_pred)
print(f"Test 3 - Poor prediction (large errors):")
print(f"  y_true: {y_true}")
print(f"  y_pred: {y_pred}")
print(f"  Score: {score:.2f}")
print(f"  Expected: Low score (<50)")
print(f"  Result: {'PASS' if score < 50 else 'FAIL'}")
print()

# Test 4: Moderate prediction
y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])
score = overlay_dx_score_simple(y_true, y_pred)
print(f"Test 4 - Moderate prediction:")
print(f"  y_true: {y_true}")
print(f"  y_pred: {y_pred}")
print(f"  Score: {score:.2f}")
print(f"  Expected: Moderate score (50-90)")
print(f"  Result: {'PASS' if 50 < score < 90 else 'FAIL'}")
print()

# Test 5: Negative values
y_true = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
y_pred = np.array([-4.8, -3.2, -0.9, 1.1, 2.9])
score = overlay_dx_score_simple(y_true, y_pred)
print(f"Test 5 - Negative values:")
print(f"  y_true: {y_true}")
print(f"  y_pred: {y_pred}")
print(f"  Score: {score:.2f}")
print(f"  Expected: High score (>80)")
print(f"  Result: {'PASS' if score > 80 else 'FAIL'}")
print()

print("=" * 60)
print("All tests completed!")
