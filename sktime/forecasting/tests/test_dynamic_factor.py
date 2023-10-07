"""Tests the DynamicFactor model."""
import pytest
from pandas.testing import assert_frame_equal

from sktime.datasets import load_longley
from sktime.forecasting.dynamic_factor import DynamicFactor
from sktime.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["yarnabrina"]

HISTORY_LENGTH = 10
PREDICTION_LENGTH = 6

K_FACTORS = 1
FACTOR_ORDER = 1

COVERAGES = [0.95, 0.99]

_, MULTIVARIATE_DF = load_longley()

TARGET_COLUMNS = ["GNPDEFL", "GNP"]
FEATURE_COLUMNS = ["UNEMP", "POP"]

ENDOGENOUS_DF = MULTIVARIATE_DF[TARGET_COLUMNS]
EXOGENOUS_DF = MULTIVARIATE_DF[FEATURE_COLUMNS]

TRAIN_Y = ENDOGENOUS_DF[:HISTORY_LENGTH]
TRAIN_X = EXOGENOUS_DF[:HISTORY_LENGTH]

PREDICT_X = EXOGENOUS_DF[HISTORY_LENGTH : (HISTORY_LENGTH + PREDICTION_LENGTH)]


def compare_predictions_against_statsmodels(
    sktime_point_predictions, sktime_interval_predictions, statsmodels_predictions
) -> None:
    """Compare predictions from ``sktime`` wrapper against ``statsmodels`` estimator.

    Notes
    -----
    * compare point predictions - predictive mean
    * compare confidence intervals for multiple coverage values, viz. ``COVERAGES``
    """
    statsmodels_point_predictions = statsmodels_predictions.predicted_mean
    assert_frame_equal(sktime_point_predictions, statsmodels_point_predictions)

    for coverage in COVERAGES:
        statsmodels_interval_predictions = statsmodels_predictions.conf_int(
            alpha=(1 - coverage)
        )

        for target in TARGET_COLUMNS:
            sktime_results = sktime_interval_predictions.xs(
                (target, coverage), axis="columns"
            )

            statsmodels_results = statsmodels_interval_predictions.filter(
                regex=f"{target}$"
            )
            statsmodels_results.columns = ["lower", "upper"]

            assert_frame_equal(sktime_results, statsmodels_results)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_DynamicFactor_without_exogenous_variables():
    """Test ``DynamicFactor`` in absence of exogenous variables."""
    from statsmodels.tsa.statespace.dynamic_factor import (
        DynamicFactor as _DynamicFactor,
    )

    unfitted_sktime_model = DynamicFactor(
        k_factors=K_FACTORS, factor_order=FACTOR_ORDER
    )
    fitted_sktime_model = unfitted_sktime_model.fit(TRAIN_Y)

    sktime_point_predictions = fitted_sktime_model.predict(
        fh=range(1, PREDICTION_LENGTH + 1)
    )
    sktime_interval_predictions = fitted_sktime_model.predict_interval(
        fh=range(1, PREDICTION_LENGTH + 1), coverage=COVERAGES
    )

    unfitted_statsmodels_model = _DynamicFactor(TRAIN_Y, K_FACTORS, FACTOR_ORDER)
    fitted_statsmodels_model = unfitted_statsmodels_model.fit()

    statsmodels_predictions = fitted_statsmodels_model.get_prediction(
        start=HISTORY_LENGTH, end=HISTORY_LENGTH + PREDICTION_LENGTH - 1
    )

    compare_predictions_against_statsmodels(
        sktime_point_predictions, sktime_interval_predictions, statsmodels_predictions
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_DynamicFactor_with_exogenous_variables():
    """Test ``DynamicFactor`` in presence of exogenous variables."""
    from statsmodels.tsa.statespace.dynamic_factor import (
        DynamicFactor as _DynamicFactor,
    )

    unfitted_sktime_model = DynamicFactor(
        k_factors=K_FACTORS, factor_order=FACTOR_ORDER
    )
    fitted_sktime_model = unfitted_sktime_model.fit(TRAIN_Y, X=TRAIN_X)

    sktime_point_predictions = fitted_sktime_model.predict(
        fh=range(1, PREDICTION_LENGTH + 1), X=PREDICT_X
    )
    sktime_interval_predictions = fitted_sktime_model.predict_interval(
        fh=range(1, PREDICTION_LENGTH + 1), X=PREDICT_X, coverage=COVERAGES
    )

    unfitted_statsmodels_model = _DynamicFactor(
        TRAIN_Y, K_FACTORS, FACTOR_ORDER, exog=TRAIN_X
    )
    fitted_statsmodels_model = unfitted_statsmodels_model.fit()

    statsmodels_predictions = fitted_statsmodels_model.get_prediction(
        start=HISTORY_LENGTH, end=HISTORY_LENGTH + PREDICTION_LENGTH - 1, exog=PREDICT_X
    )

    compare_predictions_against_statsmodels(
        sktime_point_predictions, sktime_interval_predictions, statsmodels_predictions
    )
