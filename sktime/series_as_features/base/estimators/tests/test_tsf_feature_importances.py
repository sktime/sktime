"""Tests for feature importances in time series forests."""
import numpy as np
import pandas as pd

from sktime.annotation.datagen import piecewise_normal
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.utils._testing.panel import make_classification_problem

X_train, y_train = make_classification_problem()


def test_sum_feature_importances():
    """Test feature importances sums for model classifier.

    Sum of the normalized temporal importance curves should be equal to the number of
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


def test_gen_data_importance_differences():
    """Tests synthetic data to ensure importances are where they are expected.

    Synthetic data class 0 is randomly generated with mean 0 and stdev 1.
    Synthetic data class 1 is randomly generated with mean 5 and stdev 1.
    Synthetic data class 2 is randomly generated with mean 0 and stdev 5.
    We don't check slope because the slope changes the mean and stdev in
    random intervals"""
    data0 = []
    data1 = []
    data2 = []
    n_timeseries = 30
    for _ in range(n_timeseries):
        data0.append(piecewise_normal([0], [50], [1]))
        data1.append(piecewise_normal([5], [50], [1]))
        data2.append(piecewise_normal([0], [50], [5]))

    X_train_1 = []
    X_train_2 = []
    for x in data0 + data1:
        X_train_1.append(pd.Series(x))
    for x in data0 + data2:
        X_train_2.append(pd.Series(x))
    X_train_1 = pd.DataFrame({"dim_0": X_train_1})
    X_train_2 = pd.DataFrame({"dim_0": X_train_2})

    y_train = np.pad(
        np.zeros(n_timeseries), (0, n_timeseries), "constant", constant_values=(1, 1)
    )

    n_est = 100
    clf_1 = TimeSeriesForestClassifier(n_estimators=n_est)
    clf_1.fit(X_train_1, y_train)
    clf_1.calc_temporal_curves(normalize=True)
    clf_2 = TimeSeriesForestClassifier(n_estimators=n_est)
    clf_2.fit(X_train_2, y_train)
    clf_2.calc_temporal_curves(normalize=True)

    # tests:
    # average mean importance in clf_1 is higher than in clf_2
    # average stdev imporatnce in clf_2 is higher than in clf_1

    avg_mean_1 = np.mean(clf_1.mean_curve)
    avg_mean_2 = np.mean(clf_2.mean_curve)

    avg_stdev_1 = np.mean(clf_1.stdev_curve)
    avg_stdev_2 = np.mean(clf_2.stdev_curve)

    assert avg_mean_1 > avg_mean_2
    assert avg_stdev_2 > avg_stdev_1
