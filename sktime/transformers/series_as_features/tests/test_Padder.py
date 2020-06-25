# from sklearn.pipeline import Pipeline
from sktime.datasets.base import _load_dataset
from sktime.transformers.series_as_features.truncation import \
    PaddingTransformer
# from sklearn.ensemble import RandomForestClassifier
from sktime.utils.data_container import tabularize

# import pandas as pd


def test_truncation_transformer():
    # load data
    name = 'JapaneseVowels'
    X_train, y_train = _load_dataset(name, split='train', return_X_y=True)
    X_test, y_test = _load_dataset(name, split='test', return_X_y=True)

    # print(X_train)

    padding_transformer = PaddingTransformer()
    Xt = padding_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've padded them to there normal length of 30
    data = tabularize(Xt)
    assert len(data.columns) == 30*12


def test_truncation_paramterised_transformer():
    # load data
    name = 'JapaneseVowels'
    X_train, y_train = _load_dataset(name, split='train', return_X_y=True)
    X_test, y_test = _load_dataset(name, split='test', return_X_y=True)

    # print(X_train)

    padding_transformer = PaddingTransformer(40)
    Xt = padding_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've truncated them all to (10-2) long.
    data = tabularize(Xt)
    assert len(data.columns) == 40*12
