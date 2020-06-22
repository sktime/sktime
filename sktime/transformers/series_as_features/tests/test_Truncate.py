# from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.truncation \
 import TruncationTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sktime.utils.data_container import tabularize
# import pandas as pd

from sktime.datasets.base import _load_dataset


def test_truncation_transformer():
    # load data
    name = 'PLAID'
    X_train, y_train = _load_dataset(name, split='train', return_X_y=True)
    X_test, y_test = _load_dataset(name, split='test', return_X_y=True)

    # print(X_train)

    truncated_transformer = TruncationTransformer()
    Xt = truncated_transformer.transform(X_train)

    # first length of the series in the
    # first dimension should be truncated to 101
    assert Xt[0][0].shape == (101)


test_truncation_transformer()
