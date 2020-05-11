__author__ = 'Aaron Bostrom'

from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_basic_motions
from sktime.series_as_features.compose import ColumnEnsembleClassifier



def test_heterogenous_pipeline_column_ensmbler():
    X_train, y_train = load_basic_motions("TRAIN", return_X_y=True)
    X_test, y_test = load_basic_motions("TEST", return_X_y=True)

    # dims 0-3 with alternating classifiers.
    ct = ColumnEnsembleClassifier([
            ("KNN", KNeighborsTimeSeriesClassifier(n_neighbors=1), [0]),
            ("TSF", TimeSeriesForest(), [2])
    ])
    ct.fit(X_train, y_train)
    ct.score(X_test, y_test)

