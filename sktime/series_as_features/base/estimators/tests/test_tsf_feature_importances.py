"""Tests for feature importances in time series forests."""
import numpy as np

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.utils._testing.panel import make_classification_problem

X_train, y_train = make_classification_problem()


def test_sum_feature_importances():
    """Test feature importances sums for model classifier..

    Sum of normalized temporal importance curves should be equal to number of
    estimators in ensemble.
    """
    # make classifier and fit to data
    n_est = 10
    clf = TimeSeriesForestClassifier(n_estimators=n_est)

    clf.fit(X_train, y_train)

    clf.calc_temporal_curves(normalize=True)

    mean_curve_sum = np.sum(clf.mean_curve)
    stdev_curve_sum = np.sum(clf.stdev_curve)
    slope_curve_sum = np.sum(clf.slope_curve)
    assert np.isclose(mean_curve_sum + stdev_curve_sum + slope_curve_sum, n_est)
