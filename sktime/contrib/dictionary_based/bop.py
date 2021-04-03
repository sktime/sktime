# -*- coding: utf-8 -*-
from sklearn.model_selection import GridSearchCV
from sktime.base import BaseEstimator

from sktime.classification.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesClassifier,
)
from sktime.datasets import load_italy_power_demand
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.dictionary_based import SAX
import sys
import numpy as np


class BagOfPatterns(BaseEstimator):
    __author__ = "Matthew Middlehurst"
    """ Bag of Patterns classifier
    """

    def bop_pipeline(X, y):
        steps = [
            ("transform", SAX(remove_repeat_words=True)),
            (
                "clf",
                KNeighborsTimeSeriesClassifier(
                    n_neighbors=1, distance=euclidean_distance
                ),
            ),
        ]
        pipeline = Pipeline(steps)

        series_length = X.iloc[0, 0].shape[0]
        max_window_searches = series_length / 4
        win_inc = int((series_length - 10) / max_window_searches)
        if win_inc < 1:
            win_inc = 1
        window_sizes = [win_size for win_size in range(10, series_length + 1, win_inc)]

        cv_params = {
            "transform__word_length": [8, 10, 12, 14, 16],
            "transform__alphabet_size": [2, 3, 4],
            "transform__window_size": window_sizes,
        }
        model = GridSearchCV(pipeline, cv_params, cv=5)
        model.fit(X, y)
        return model


def euclidean_distance(first, second, best_dist=sys.float_info.max):
    dist = 0

    if isinstance(first, dict):
        words = set(list(first) + list(second))
        for word in words:
            val_a = first.get(word, 0)
            val_b = second.get(word, 0)
            dist += (val_a - val_b) * (val_a - val_b)

            if dist > best_dist:
                return sys.float_info.max
    else:
        dist = np.sum(
            [(first[n] - second[n]) * (first[n] - second[n]) for n in range(len(first))]
        )

    return dist


if __name__ == "__main__":
    X_train, y_train = load_italy_power_demand(split="TRAIN", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="TEST", return_X_y=True)

    model = bop_pipeline(X_train, y_train)
    model.predict(X_test)
