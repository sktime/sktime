
from sktime.pipeline import FeatureUnion, Pipeline
from sklearn.tree import DecisionTreeClassifier
from sktime.transformers.series_to_series import RandomIntervalSegmenter
from sktime.transformers.series_to_tabular import RandomIntervalFeatureExtractor
from sklearn.preprocessing import FunctionTransformer
from sktime.transformers.compose import RowwiseTransformer

import numpy as np

from sklearn.neighbors.classification import KNeighborsClassifier
from sktime.classifiers.time_series_neighbors import KNeighborsTimeSeriesClassifier as KNNTSC
from sktime.datasets.base import _load_dataset
from sktime.contrib.dictionary_based.boss_ensemble import BOSSEnsemble
from sktime. contrib.column_ensembler import ColumnEnsembler, SimpleColumnEnsembler


def test_univariate_column_ensembler():
    ct = ColumnEnsembler(
    [("KNN1", KNNTSC(n_neighbors=1), [1]),
    ("KNN2", KNNTSC(n_neighbors=1), [2])]
    )


def test_simple_column_ensembler():
    X_train ,y_train  = _load_dataset("JapaneseVowels", "TRAIN", True)
    X_test, y_test = _load_dataset("JapaneseVowels", "TEST", True)

    cts = SimpleColumnEnsembler(KNNTSC(n_neighbors=1))

    cts.fit(X_train, y_train)
    print(cts.score(X_train, y_train))



def test_homogeneous_pipeline_column_ensmbler():

    X_train ,y_train  = _load_dataset("JapaneseVowels", "TRAIN", True)
    X_test, y_test = _load_dataset("JapaneseVowels", "TEST", True)

    ct = ColumnEnsembler(
        [("KNN%d " %i, KNNTSC(n_neighbors=1), [i]) for i in range(0 ,X_train.shape[1])]
    )

    ct.fit(X_train, y_train)
    print(ct.score(X_train,y_test))

def test_heterogenous_pipeline_column_ensmbler():
    X_train, y_train = _load_dataset("JapaneseVowels", "TRAIN", True)
    X_test, y_test = _load_dataset("JapaneseVowels", "TEST", True)

    n_intervals=3

    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals=n_intervals, check_input=False)),
        ('transform', FeatureUnion([
            ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
            ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False)))
        ])),
        ('clf', DecisionTreeClassifier())
    ]
    clf1 = Pipeline(steps, random_state=1)


    #dims 0-3 with alternating classifiers.
    ct = ColumnEnsembler(
        [
            ("RandomIntervalTree", clf1, [0]),
            ("KNN4", KNNTSC(n_neighbors=1), [4]),
            ("BOSSEnsemble1 ", BOSSEnsemble(), [1]),
            ("KNN2", KNNTSC(n_neighbors=1), [2]),
            ("BOSSEnsemble3", BOSSEnsemble(), [3]),
         ]
    )

    ct.fit(X_train, y_train)
    print(ct.score(X_train, y_test))


if __name__ == "__main__":
    test_heterogenous_pipeline_column_ensmbler()