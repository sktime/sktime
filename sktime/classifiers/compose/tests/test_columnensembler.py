__author__ = 'Aaron Bostrom'

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer

from sktime.pipeline import FeatureUnion, Pipeline
from sktime.transformers.segment import RandomIntervalSegmenter
from sktime.transformers.compose import RowwiseTransformer
from sktime.classifiers.distance_based import KNeighborsTimeSeriesClassifier as KNNTSC
from sktime.datasets import load_basic_motions
from sktime.classifiers.dictionary_based import BOSSEnsemble
from sktime.classifiers.compose import ColumnEnsembleClassifier, HomogeneousColumnEnsembleClassifier


def test_univariate_column_ensembler_init():
    ct = ColumnEnsembleClassifier(
        [("KNN1", KNNTSC(n_neighbors=1), [1]),
         ("KNN2", KNNTSC(n_neighbors=1), [2])]
    )


def test_homogeneous_column_ensembler():
    X_train, y_train = load_basic_motions("TRAIN", return_X_y=True)
    X_test, y_test = load_basic_motions("TEST", return_X_y=True)

    cts = HomogeneousColumnEnsembleClassifier(KNNTSC(n_neighbors=1))

    cts.fit(X_train, y_train)
    cts.score(X_test, y_test) == 1.0


def test_homogeneous_pipeline_column_ensmbler():
    X_train, y_train = load_basic_motions("TRAIN", return_X_y=True)
    X_test, y_test = load_basic_motions("TEST", return_X_y=True)

    ct = ColumnEnsembleClassifier(
        [("KNN%d " % i, KNNTSC(n_neighbors=1), [i]) for i in range(0, X_train.shape[1])]
    )

    ct.fit(X_train, y_train)
    ct.score(X_test, y_test)


def test_heterogenous_pipeline_column_ensmbler():
    X_train, y_train = load_basic_motions("TRAIN", return_X_y=True)
    X_test, y_test = load_basic_motions("TEST", return_X_y=True)

    n_intervals = 3

    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals=n_intervals)),
        ('transform', FeatureUnion([
            ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
            ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False)))
        ])),
        ('clf', DecisionTreeClassifier())
    ]
    clf1 = Pipeline(steps, random_state=1)

    # dims 0-3 with alternating classifiers.
    ct = ColumnEnsembleClassifier(
        [
            ("RandomIntervalTree", clf1, [0]),
            ("KNN4", KNNTSC(n_neighbors=1), [4]),
            ("BOSSEnsemble1 ", BOSSEnsemble(n_parameter_samples=3), [1]),
            ("KNN2", KNNTSC(n_neighbors=1), [2]),
            ("BOSSEnsemble3", BOSSEnsemble(n_parameter_samples=3), [3]),
        ]
    )

    ct.fit(X_train, y_train)
    ct.score(X_test, y_test)


if __name__ == "__main__":
    test_heterogenous_pipeline_column_ensmbler()
