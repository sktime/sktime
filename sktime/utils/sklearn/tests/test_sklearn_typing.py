"""Tests for sklearn typing utilities in utils.sktime."""

__author__ = ["fkiraly"]


import pytest
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from sktime.classification.feature_based import SummaryClassifier
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.sklearn import is_sklearn_estimator, sklearn_scitype

CORRECT_SCITYPES = {
    KMeans: "clusterer",
    KNeighborsClassifier: "classifier",
    KNeighborsRegressor: "regressor",
    StandardScaler: "transformer",
}

sklearn_estimators = list(CORRECT_SCITYPES.keys())
sktime_estimators = [SummaryClassifier, NaiveForecaster]


@pytest.mark.skipif(
    not run_test_for_class(is_sklearn_estimator),
    reason="Run if utilities have changed.",
)
@pytest.mark.parametrize("estimator", sklearn_estimators)
def test_is_sklearn_estimator_positive(estimator):
    """Test that is_sklearn_estimator recognizes positive examples correctly."""
    msg = (
        f"is_sklearn_estimator incorrectly considers {estimator.__name__} "
        f"as not an sklearn estimator (output False), but output should be True"
    )
    assert is_sklearn_estimator(estimator), msg


@pytest.mark.skipif(
    not run_test_for_class(is_sklearn_estimator),
    reason="Run if utilities have changed.",
)
@pytest.mark.parametrize("estimator", sktime_estimators)
def test_is_sklearn_estimator_negative(estimator):
    """Test that is_sklearn_estimator recognizes negative examples correctly."""
    msg = (
        f"is_sklearn_estimator incorrectly considers {estimator.__name__} "
        f"as an sklearn estimator (output True), but output should be False"
    )
    assert not is_sklearn_estimator(estimator), msg


@pytest.mark.skipif(
    not run_test_for_class(sklearn_scitype),
    reason="Run if utilities have changed.",
)
@pytest.mark.parametrize("estimator", sklearn_estimators)
def test_sklearn_scitype(estimator):
    """Test that sklearn_scitype returns the correct scitype string."""
    scitype = sklearn_scitype(estimator)
    expected_scitype = CORRECT_SCITYPES[estimator]
    msg = (
        f"is_sklearn_estimator returns the incorrect scitype string for "
        f'"{estimator.__name__}". Should be {expected_scitype}, but '
        f'{scitype}" was returned.'
    )
    assert scitype == expected_scitype, msg
