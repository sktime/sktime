"""Tests for probabilistic performance metrics."""

import warnings

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.forecasting.probabilistic import (
    ConstraintViolation,
    EmpiricalCoverage,
    IntervalWidth,
    PinballLoss,
)
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.series import _make_series

warnings.filterwarnings("ignore", category=FutureWarning)

quantile_metrics = [
    PinballLoss,
]

interval_metrics = [
    ConstraintViolation,
    EmpiricalCoverage,
    IntervalWidth,
]

all_metrics = interval_metrics + quantile_metrics

alpha_s = [0.5]
alpha_m = [0.05, 0.5, 0.95]
coverage_s = 0.9
coverage_m = [0.7, 0.8, 0.9, 0.99]


@pytest.fixture
def sample_data(request):
    n_columns, coverage_or_alpha, pred_type = request.param

    y = _make_series(n_columns=n_columns)
    y_train, y_test = temporal_train_test_split(y)

    if pred_type == "interval":
        interval_pred = {}
        for col in range(n_columns):
            if n_columns == 1:
                y_vals = y_test
            else:
                y_vals = y_test.iloc[:, col]
            for coverage in np.array([coverage_or_alpha]).flatten():
                interval_pred[(col, coverage, "lower")] = (
                    y_vals * np.random.uniform(0.9, 1.1, len(y_vals)) * (1 - coverage)
                )
                interval_pred[(col, coverage, "upper")] = (
                    y_vals * np.random.uniform(0.9, 1.1, len(y_vals)) * (1 + coverage)
                )
        interval_pred = pd.DataFrame.from_dict(interval_pred)
        return y_test, interval_pred

    elif pred_type == "quantile":
        quantile_pred = {}
        for col in range(n_columns):
            if n_columns == 1:
                y_vals = y_test
            else:
                y_vals = y_test.iloc[:, col]
            for alpha in coverage_or_alpha:
                quantile_pred[(col, alpha)] = (
                    y_vals * np.random.uniform(0.9, 1.1, len(y_vals)) * (0.5 + alpha)
                )
        quantile_pred = pd.DataFrame.from_dict(quantile_pred)
        return y_test, quantile_pred

    return


# Test the parametrized fixture
@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize(
    "sample_data",
    [
        (1, alpha_s, "quantile"),
        (3, alpha_s, "quantile"),
        (1, alpha_m, "quantile"),
        (3, alpha_m, "quantile"),
        (1, coverage_s, "interval"),
        (3, coverage_s, "interval"),
        (1, coverage_m, "interval"),
        (3, coverage_m, "interval"),
    ],
    indirect=True,
)
def test_sample_data(sample_data):
    y_true, y_pred = sample_data
    assert isinstance(y_true, (pd.Series, pd.DataFrame))
    assert isinstance(y_pred, pd.DataFrame)


def helper_check_output(metric, score_average, multioutput, sample_data):
    y_true, y_pred = sample_data
    """Test output is correct class and shape for given data."""
    loss = metric.create_test_instance()
    loss.set_params(score_average=score_average, multioutput=multioutput)

    eval_loss = loss(y_true, y_pred)
    index_loss = loss.evaluate_by_index(y_true, y_pred)

    no_vars = len(y_pred.columns.get_level_values(0).unique())
    no_scores = len(y_pred.columns.get_level_values(1).unique())

    if (
        0.5 in y_pred.columns.get_level_values(1)
        and loss.get_tag("scitype:y_pred") == "pred_interval"
        and y_pred.columns.nlevels == 2
    ):
        no_scores = no_scores - 1
        no_scores = no_scores / 2  # one interval loss per two quantiles given
        if no_scores == 0:  # if only 0.5 quant, no output to interval loss
            no_vars = 0

    if score_average and multioutput == "uniform_average":
        assert isinstance(eval_loss, float)
        assert isinstance(index_loss, pd.Series)

        assert len(index_loss) == y_pred.shape[0]

    if not score_average and multioutput == "uniform_average":
        assert isinstance(eval_loss, pd.Series)
        assert isinstance(index_loss, pd.DataFrame)

        # get two quantiles from each interval so if not score averaging
        # get twice number of unique coverages
        if (
            loss.get_tag("scitype:y_pred") == "pred_quantiles"
            and y_pred.columns.nlevels == 3
        ):
            assert len(eval_loss) == 2 * no_scores
        else:
            assert len(eval_loss) == no_scores

    if not score_average and multioutput == "raw_values":
        assert isinstance(eval_loss, pd.Series)
        assert isinstance(index_loss, pd.DataFrame)

        true_len = no_vars * no_scores

        if (
            loss.get_tag("scitype:y_pred") == "pred_quantiles"
            and y_pred.columns.nlevels == 3
        ):
            assert len(eval_loss) == 2 * true_len
        else:
            assert len(eval_loss) == true_len

    if score_average and multioutput == "raw_values":
        assert isinstance(eval_loss, pd.Series)
        assert isinstance(index_loss, pd.DataFrame)

        assert len(eval_loss) == no_vars


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize(
    "sample_data",
    [
        (1, alpha_s, "quantile"),
        (3, alpha_s, "quantile"),
        (1, alpha_m, "quantile"),
        (3, alpha_m, "quantile"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("metric", all_metrics)
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
@pytest.mark.parametrize("score_average", [True, False])
def test_output_quantiles(metric, score_average, multioutput, sample_data):
    helper_check_output(metric, score_average, multioutput, sample_data)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize(
    "sample_data",
    [
        (1, coverage_s, "interval"),
        (3, coverage_s, "interval"),
        (1, coverage_m, "interval"),
        (3, coverage_m, "interval"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("metric", all_metrics)
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
@pytest.mark.parametrize("score_average", [True, False])
def test_output_intervals(metric, score_average, multioutput, sample_data):
    helper_check_output(metric, score_average, multioutput, sample_data)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("metric", quantile_metrics)
@pytest.mark.parametrize(
    "sample_data",
    [
        (1, alpha_s, "quantile"),
        (3, alpha_s, "quantile"),
        (1, alpha_m, "quantile"),
        (3, alpha_m, "quantile"),
    ],
    indirect=True,
)
def test_evaluate_alpha_positive(metric, sample_data):
    """Tests output when required quantile is present."""
    # 0.5 in test quantile data don't raise error.

    y_true, y_pred = sample_data

    Loss = metric.create_test_instance().set_params(alpha=0.5, score_average=False)
    res = Loss(y_true=y_true, y_pred=y_pred)
    assert len(res) == 1

    if all(x in y_pred.columns.get_level_values(1) for x in [0.5, 0.95]):
        Loss = metric.create_test_instance().set_params(
            alpha=[0.5, 0.95], score_average=False
        )
        res = Loss(y_true=y_true, y_pred=y_pred)
        assert len(res) == 2


# This test tests quantile data
@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize(
    "sample_data",
    [
        (1, alpha_s, "quantile"),
        (3, alpha_s, "quantile"),
        (1, alpha_m, "quantile"),
        (3, alpha_m, "quantile"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("metric", quantile_metrics)
def test_evaluate_alpha_negative(metric, sample_data):
    """Tests whether correct error raised when required quantile not present."""
    y_true, y_pred = sample_data
    with pytest.raises(ValueError):
        # 0.3 not in test quantile data so raise error.
        Loss = metric.create_test_instance().set_params(alpha=0.3)
        res = Loss(y_true=y_true, y_pred=y_pred)  # noqa: F841


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize("metric", all_metrics)
@pytest.mark.parametrize("score_average", [True, False])
def test_multioutput_weighted(metric, score_average):
    """Test output contracts for multioutput weights."""
    y_true = pd.DataFrame({"var1": [3, -0.5, 2, 7, 2], "var2": [4, 0.5, 3, 8, 3]})
    y_pred = pd.DataFrame(
        {
            ("var1", 0.05): [1.5, -1, 1, 4, 0.65],
            ("var1", 0.5): [2.5, 0, 2, 8, 1.25],
            ("var1", 0.95): [3.5, 4, 3, 12, 1.85],
            ("var2", 0.05): [2.5, 0, 2, 8, 1.25],
            ("var2", 0.5): [5.0, 1, 4, 16, 2.5],
            ("var2", 0.95): [7.5, 2, 6, 24, 3.75],
        }
    )

    weights = np.array([0.3, 0.7])

    loss = metric.create_test_instance()
    loss.set_params(score_average=score_average, multioutput=weights)

    eval_loss = loss(y_true, y_pred)

    if loss.get_tag("scitype:y_pred") == "pred_interval":
        # 1 full interval, lower = 0.05, upper = 0.95
        expected_score_ix = [0.9]
    else:
        # 3 quantile scores, 0.05, 0.5, 0.95
        expected_score_ix = [0.05, 0.5, 0.95]
    no_expected_scores = len(expected_score_ix)
    expected_timepoints = len(y_pred)

    if score_average:
        assert isinstance(eval_loss, float)
    else:
        assert isinstance(eval_loss, pd.Series)
        assert len(eval_loss) == no_expected_scores

    eval_loss_by_index = loss.evaluate_by_index(y_true, y_pred)
    assert len(eval_loss_by_index) == expected_timepoints

    if score_average:
        assert isinstance(eval_loss_by_index, pd.Series)
    else:
        assert isinstance(eval_loss_by_index, pd.DataFrame)
        assert eval_loss_by_index.shape == (expected_timepoints, no_expected_scores)
        assert eval_loss_by_index.columns.to_list() == expected_score_ix


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
@pytest.mark.parametrize(
    "y_true_vals, lower_vals, upper_vals",
    [
        ([2.0], [2.0], [3.0]),  # True value equals lower boundary
        ([3.0], [2.0], [3.0]),  # True value equals upper boundary
        ([2.0, 3.0], [2.0, 2.0], [3.0, 3.0]),  # Both boundaries
    ],
)
def test_empirical_coverage_boundary_equality(y_true_vals, lower_vals, upper_vals):
    """Test EmpiricalCoverage when true values equal interval boundaries.

    This test ensures that when a true value equals the lower or upper
    boundary of a prediction interval, it is correctly counted as covered
    (i.e., uses non-strict inequalities).
    """
    y_true = pd.Series(y_true_vals)
    y_pred = pd.DataFrame(
        {
            ("var1", 0.8, "lower"): lower_vals,
            ("var1", 0.8, "upper"): upper_vals,
        }
    )

    metric = EmpiricalCoverage()
    coverage = metric(y_true, y_pred)

    assert coverage == 1.0, (
        f"Expected coverage 1.0 but got {coverage}. "
        "True values on interval boundaries should be counted as covered."
    )


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.performance_metrics"]),
    reason="Run if performance_metrics module has changed.",
)
def test_univariate_y_true_multivariate_y_pred_issue_11324():
    """Regression test for Issue #11324: univariate y_true + multivariate MultiIndex y_pred.

    The fix in _check_ys broadcasts y_true to match y_pred's variables.
    This test verifies correct variable-wise alignment with deterministic data.
    """
    # Univariate y_true (single Series)
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    # Multivariate y_pred with 2 variables (B, A) and 3 quantiles each
    # Using deterministic values so we can verify exact results
    # B comes before A to test non-alphabetical variable names (Issue #11324)
    cols = pd.MultiIndex.from_product(
        [["B", "A"], [0.1, 0.5, 0.9]],
        names=["var", "q"],
    )
    # y_pred values designed so pinball loss per quantile is predictable
    # For y_true=[1,2,3,4,5], quantile predictions are y_true +/- offset
    # B data: [0.5, 1.0, 1.5, ...], A data: [0.8, 1.3, 1.8, ...]
    y_pred_data = np.array([
        # B@0.1  B@0.5  B@0.9  A@0.1  A@0.5  A@0.9
        [0.5, 1.0, 1.5, 0.8, 1.3, 1.8],  # t=1
        [1.5, 2.0, 2.5, 1.8, 2.3, 2.8],  # t=2
        [2.5, 3.0, 3.5, 2.8, 3.3, 3.8],  # t=3
        [3.5, 4.0, 4.5, 3.8, 4.3, 4.8],  # t=4
        [4.5, 5.0, 5.5, 4.8, 5.3, 5.8],  # t=5
    ])
    y_pred = pd.DataFrame(y_pred_data, columns=cols)

    # --- PinballLoss: score_average=False, multioutput="raw_values" ---
    # Returns per-variable-per-quantile scores (2 variables × 3 quantiles = 6 values)
    # Expected per-quantile losses: B: [0.05, 0.00, 0.05], A: [0.02, 0.15, 0.08]
    metric = PinballLoss(score_average=False, multioutput="raw_values")
    score = metric(y_true, y_pred)

    assert isinstance(score, pd.Series)
    assert len(score) == 6
    assert score.index.names == ["var", "q"]
    # The order will be sorted by variable name: A, B (A: 0.02, 0.15, 0.08, B: 0.05, 0.00, 0.05)
    np.testing.assert_allclose(
        score.to_numpy(),
        np.array([0.02, 0.15, 0.08, 0.05, 0.00, 0.05]),
        rtol=1e-10,
    )
    # Check index order (should be sorted A, B)
    assert score.index.levels[0].tolist() == ["A", "B"]

    # --- PinballLoss: score_average=True, multioutput="uniform_average" ---
    # Averages over quantiles AND variables
    # Expected: mean(mean(0.02, 0.15, 0.08), mean(0.05, 0.00, 0.05)) = mean(0.08333, 0.03333) = 0.058333...
    metric_unif = PinballLoss(score_average=True, multioutput="uniform_average")
    score_unif = metric_unif(y_true, y_pred)
    assert isinstance(score_unif, float)
    np.testing.assert_allclose(score_unif, 7.0 / 120.0, rtol=1e-10)

    # --- PinballLoss: score_average=True, multioutput="raw_values" ---
    # Per-variable average over quantiles
    metric_raw = PinballLoss(score_average=True, multioutput="raw_values")
    score_raw = metric_raw(y_true, y_pred)
    assert isinstance(score_raw, pd.Series)
    assert len(score_raw) == 2
    # Sorted order: A: 1/12, B: 1/30
    np.testing.assert_allclose(
        score_raw.to_numpy(),
        np.array([1.0 / 12.0, 1.0 / 30.0]),
        rtol=1e-10,
    )

    # --- EmpiricalCoverage ---
    # Create interval predictions where y_true is always inside the interval
    # Sorted order: A, B
    interval_cols = pd.MultiIndex.from_product(
        [["A", "B"], [0.8], ["lower", "upper"]],
        names=["var", "cov", "bound"],
    )
    # y_true in [1,2,3,4,5]; intervals [0.5,1.5], [1.5,2.5], etc. always cover
    # B: [0.5,1.5], A: [0.8,1.8]
    interval_data = np.array([
        [0.8, 1.8, 0.5, 1.5],
        [1.8, 2.8, 1.5, 2.5],
        [2.8, 3.8, 2.5, 3.5],
        [3.8, 4.8, 3.5, 4.5],
        [4.8, 5.8, 4.5, 5.5],
    ])
    y_pred_interval = pd.DataFrame(interval_data, columns=interval_cols)

    # score_average=True, multioutput="raw_values" -> per-variable coverage
    metric_cov_raw = EmpiricalCoverage(score_average=True, multioutput="raw_values")
    coverage_raw = metric_cov_raw(y_true, y_pred_interval)
    assert isinstance(coverage_raw, pd.Series)
    assert len(coverage_raw) == 2
    np.testing.assert_allclose(coverage_raw.to_numpy(), np.array([1.0, 1.0]))

    # score_average=True, multioutput="uniform_average" -> scalar
    metric_cov_unif = EmpiricalCoverage(score_average=True, multioutput="uniform_average")
    coverage_unif = metric_cov_unif(y_true, y_pred_interval)
    assert coverage_unif == 1.0

    # --- ConstraintViolation ---
    # Same intervals, y_true always inside => violation = 0
    metric_cv_raw = ConstraintViolation(score_average=True, multioutput="raw_values")
    violation_raw = metric_cv_raw(y_true, y_pred_interval)
    assert isinstance(violation_raw, pd.Series)
    assert len(violation_raw) == 2
    np.testing.assert_allclose(violation_raw.to_numpy(), np.array([0.0, 0.0]))

    metric_cv_unif = ConstraintViolation(score_average=True, multioutput="uniform_average")
    violation_unif = metric_cv_unif(y_true, y_pred_interval)
    assert violation_unif == 0.0
