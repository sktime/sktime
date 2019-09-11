__author__ = "Markus Löning"

import numpy as np
import pandas as pd


def feature_importances(tsf):
    """Compute feature importances for time series forest classifier
    """
    # assumes particular structure of clf, with each tree consisting of a particular pipeline, as in modular tsf

    # get series length, assuming same length series
    tree = tsf.estimators_[0]
    transformer = tree.steps[0][1]
    time_index = transformer._time_index
    n_timepoints = len(time_index)

    # get feature names, features are the same for all trees
    feature_names = [feature.__name__ for feature in transformer.features]
    n_features = len(feature_names)

    # get intervals from transformer, the number of intervals is the same for all trees
    intervals = transformer.intervals_
    n_intervals = len(intervals)

    # get number of estimators
    n_estimators = len(tsf.estimators_)

    # preallocate array for feature importances
    fis = np.zeros((n_timepoints, n_features))

    for i in range(n_estimators):
        # select tree
        tree = tsf.estimators_[i]
        transformer = tree.steps[0][1]
        classifier = tree.steps[-1][1]

        # get intervals from transformer
        intervals = transformer.intervals_

        # get feature importances from classifier
        fi = classifier.feature_importances_

        for k in range(n_features):
            for j in range(n_intervals):
                # get start and end point from interval
                start, end = intervals[j]

                # get time index for interval
                interval_time_points = np.arange(start, end)

                # get index for feature importances, assuming particular order of features
                column_index = (k * n_intervals) + j

                # add feature importance for all time points of interval
                fis[interval_time_points, k] += fi[column_index]

    # normalise by number of estimators and number of intervals
    fis = fis / n_estimators / n_intervals

    # format output
    fis = pd.DataFrame(fis, columns=feature_names, index=time_index)
    return fis
