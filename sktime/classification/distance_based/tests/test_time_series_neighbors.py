# -*- coding: utf-8 -*-

from sktime.classification.distance_based._time_series_neighbors import(
    KNeighborsTimeSeriesClassifier,
)
from sktime.datasets import load_arrow_head

distance_functions = [
    "dtw",
    "msm",
]

# expected correct on test set using default parameters. Verified in tsml
expected_correct = {
    "dtw": 123,
    "msm": 139,
}


def test_knn_on_arrowhead():
    # load gunpoint data
    X_train, y_train = load_arrow_head(split="train", return_X_y=True)
    X_test, y_test = load_arrow_head(split="test", return_X_y=True)
    for i in range(0, len(distance_functions)):
        knn = KNeighborsTimeSeriesClassifier(
            metric=distance_functions[i],
        )
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        correct = 0
        for j in range(0, len(pred)):
            if pred[j] == y_test[j]:
                correct = correct + 1
        assert correct == expected_correct[distance_functions[i]]
