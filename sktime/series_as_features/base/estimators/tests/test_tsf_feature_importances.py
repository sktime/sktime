"""Tests for feature importances in time series forests."""
import numpy as np
import pandas as pd

from sktime.annotation.datagen import piecewise_normal
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


def test_gen_data_importance_differences():
    """Tests synthetic data to ensure importances are where they are expected.

    Synthetic data class 0 is randomly generated with mean 0 and stdev 1.
    Synthetic data class 1 is segmented into 7 parts:
        Part 1 has mean 0 and stdev 1.
        Part 2 has mean 3 and stdev 1.
        Part 3 has mean 0 and stdev 1.
        Part 4 has mean 0 and stdev 3.
        Part 5 has mean 0 and stdev 1.
        Part 6 has mean 0 and stdev 1, but is shifted so that its avg slope is 1.
            This also changes the stdev of this segement away from the generated value.
        Part 7 has mean 0 and stdev 1.

    The three tests check that the average values of the importances in the
    defined regions are higher than elsewhere."""
    data1 = []
    data2 = []
    slope_offset = np.linspace(-5, 5, 10)
    slope_offset = np.pad(slope_offset, (50, 10), "constant", constant_values=(0, 0))
    for i in range(30):
        data1.append(piecewise_normal([0], [70], [1]))
        data2.append(
            piecewise_normal([0, 3, 0, 0, 0, 0, 0], [10] * 7, [1, 1, 1, 3, 1, 1, 1])
        )
        data2[i] += slope_offset

    X_train = []
    for x in data1 + data2:
        X_train.append(pd.Series(x))
    X_train = pd.DataFrame({"dim_0": X_train})

    y_train = np.pad(np.zeros(30), (0, 30), "constant", constant_values=(1, 1))

    n_est = 100
    clf = TimeSeriesForestClassifier(n_estimators=n_est)
    clf.fit(X_train, y_train)
    clf.calc_temporal_curves(normalize=True)

    # tests:
    # average mean importance in [10,19] > ([0,9],[20,69])
    # average slope imporatnce in [50,59] > ([0,49],[60,69])
    # average stdev imporatnce in ([30,39],[50,59]) > ([0,29],[40,49],[60,69])

    avg_mean_int = np.mean(clf.mean_curve[10:19])
    avg_mean_ext = np.mean(clf.mean_curve[0:9]) + np.mean(clf.mean_curve[20:69])

    avg_slope_int = np.mean(clf.slope_curve[50:59])
    avg_slope_ext = np.mean(clf.slope_curve[0:49]) + np.mean(clf.slope_curve[60:69])

    avg_stdev_int = np.mean(clf.stdev_curve[30:39]) + np.mean(clf.stdev_curve[50:59])
    avg_stdev_ext = (
        np.mean(clf.stdev_curve[0:29])
        + np.mean(clf.stdev_curve[40:49])
        + np.mean(clf.stdev_curve[60:69])
    )

    assert avg_mean_int > avg_mean_ext  # mean
    assert avg_slope_int > avg_slope_ext  # slope
    assert avg_stdev_int > avg_stdev_ext  # stdev
