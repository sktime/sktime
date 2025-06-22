import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile._optimal import (
    NonNegativeOptimalReconciler,
    OptimalReconciler,
    _create_summing_matrix_from_index,
)
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.fixture
def small_hier_index():
    """
    A small MultiIndex hierarchy for testing.
    We'll have a few aggregator levels and a few bottom (leaf) series.
    """
    tuples = [
        # bottom-level nodes (no '__total')
        ("regionA", "storeA", "catA", "deptA"),
        ("regionA", "storeA", "catA", "deptB"),
        ("regionA", "storeA", "catB", "deptC"),
        ("regionB", "storeC", "catA", "deptA"),
        # aggregator nodes
        ("regionA", "storeA", "catA", "__total"),
        ("regionA", "storeA", "catB", "__total"),
        ("regionA", "storeA", "__total", "__total"),
        ("regionB", "storeC", "__total", "__total"),
        ("__total", "__total", "__total", "__total"),
    ]
    names = ["region", "store", "category", "department"]
    return pd.MultiIndex.from_tuples(tuples, names=names)


def test_create_summing_matrix_from_index(small_hier_index):
    # Given our small index, let's compute the summation matrix
    S_df = _create_summing_matrix_from_index(small_hier_index)

    # --- 1) Basic checks ---
    # We expect:
    #   N = total nodes = 9
    #   M = bottom nodes (no '__total') = 4
    assert S_df.shape == (9, 4), "Summation matrix shape should be (9 x 4)."

    # --- 2) Check that each aggregator row sums to the number of bottom nodes
    # it covers ---
    # Let's pick a known aggregator: ('regionA', 'storeA', 'catA', '__total')
    # This should be an ancestor of the bottom nodes that start with
    # ('regionA', 'storeA', 'catA', ...)
    row_agg = ("regionA", "storeA", "catA", "__total")
    # The corresponding bottom nodes are:
    # ('regionA', 'storeA', 'catA', 'deptA') and ('regionA', 'storeA', 'catA',
    #  'deptB')
    # so we expect 2 ones in that row
    expected_sum = 2
    actual_sum = S_df.loc[row_agg].sum()
    assert actual_sum == expected_sum, (
        f"Row for {row_agg} should sum to {expected_sum}, got {actual_sum}"
    )

    # --- 3) Check that a fully aggregated node has 1 for all bottom nodes ---
    # e.g., the global aggregator ('__total', '__total', '__total', '__total')
    global_agg = ("__total", "__total", "__total", "__total")
    # Should be ancestor of all 4 bottom-level nodes
    global_sum_expected = 4
    global_sum_actual = S_df.loc[global_agg].sum()
    assert global_sum_actual == global_sum_expected, (
        f"Global aggregator row should sum to {global_sum_expected}"
        f"got {global_sum_actual}"
    )

    # --- 4) Check that bottom-level rows (leaf series) have a 1 only in
    # their own column ---
    # For example, ('regionA', 'storeA', 'catA', 'deptA') should have a 1 in
    # column ('regionA', 'storeA', 'catA', 'deptA') and 0 elsewhere.
    leaf_node = ("regionA", "storeA", "catA", "deptA")
    row_values = S_df.loc[leaf_node].values
    # Expect exactly one "1" in the matching column, and zeros elsewhere
    assert np.count_nonzero(row_values) == 1, (
        f"Leaf node {leaf_node} should have exactly 1 in its row,"
        f" got {np.count_nonzero(row_values)}"
    )
    # Check that the position of the 1 is exactly the matching column
    leaf_node_col_index = S_df.columns.tolist().index(leaf_node)
    assert row_values[leaf_node_col_index] == 1, (
        "Leaf node row should have a 1 in its own column."
    )

    # If all assertions pass, the test passes
    print("All tests passed for create_summing_matrix_from_index!")


@pytest.mark.skipif(
    not run_test_for_class(NonNegativeOptimalReconciler),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("hierarchical_levels", [(2, 4), (3, 3), (2, 2, 2)])
def test_nonnegative_reconciliation(hierarchical_levels):
    y = _make_hierarchical(
        hierarchy_levels=hierarchical_levels, max_timepoints=12, min_timepoints=12
    )

    y = Aggregator().fit_transform(y)
    y *= 0
    # Add noise
    y += np.random.normal(0, 1, y.shape)

    reconciler = NonNegativeOptimalReconciler()
    reconciler.fit(y)
    yreconc = reconciler.inverse_transform(y)
    assert np.all(yreconc >= 0), "Negative values in reconciled series!"


@pytest.mark.parametrize(
    "W",
    [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 5]]),
        np.array([[1, 0.2, 0.3], [0.2, 5, 0.6], [0.3, 0.6, 9]]),
    ],
)
@pytest.mark.parametrize(
    "Reconciler", [OptimalReconciler, NonNegativeOptimalReconciler]
)
def test_mint_kills_weighted_rowspace_perturbation(W, Reconciler):
    """
    Test if a pertubation is cancelled by the MinT reconciler.

    In the zero-constrained approach, we reconciled by computing the matrix M:

    y_reconc = M @ y_base

    where M =  I - WC'(CWC')^(-1)C, and C is the constrain matrix such that
    C@y = 0 for all coherent y

    Then, if y_base = (y_tilde + WC'), for y_tilde such that C @ y_tilde = 0
    M @ y_base = M @ (y_tilde + WC') = M @ y_tilde + M @ WC'
    = y_tilde + (I - WC'(CWC')^(-1)C) @ WC') = y_tilde + (WC' - WC') = y_tilde
    """

    if Reconciler == NonNegativeOptimalReconciler and not _check_soft_dependencies(
        "cvxpy", severity="none"
    ):
        pytest.skip("NonNegativeOptimalReconciler requires cvxpy, skipping test.")

    idx = pd.MultiIndex.from_tuples(
        [
            ("__total", "__total", 0),
            ("parent", "child1", 0),
            ("parent", "child2", 0),
        ],
        names=["lvl1", "lvl2", "time"],
    )
    y = pd.DataFrame({"value": [5.0, 2.0, 3.0]}, index=idx)

    W_df = pd.DataFrame(
        data=W,
        index=y.index.droplevel(-1),
        columns=y.index.droplevel(-1),
    )
    rec = Reconciler(error_covariance_matrix=W_df, alpha=0.0)
    base = rec.fit_transform(y)  # identity transform for base forecasts

    # hand-construct the single constraint C = [1, -1, -1]
    #   so that Cy = 0 iff total = c1 + c2
    C = np.array([[1, -1, -1]])  # shape (1,3)

    # a perturbation in the row-space of W C^T,
    # which the weighted projector must annihilate:
    v = W @ C.T

    v = v / v.max()
    # add it to the base forecasts
    yt = base.copy()
    yt["value"] = yt["value"].values + v.ravel()
    y_rec = rec.inverse_transform(yt)
    np.testing.assert_allclose(
        y.values,  # original
        y_rec.values,  # after adding v then reconciling
        atol=1e-8,
        err_msg="MinT did not kill the WC' perturbation!",
    )
    # and the index must be preserved
    assert (y.index == y_rec.index).all()
