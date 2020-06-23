# from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.truncation \
 import TruncationTransformer
# from sklearn.ensemble import RandomForestClassifier
from sktime.utils.data_container import tabularize
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

    # we should be able to tabularise this data as its noq square.
    # in addition the shortest series in PLAID is 101 long
    # so it should have 101 cols
    data = tabularize(Xt)
    assert len(data.columns) == 101


test_truncation_transformer()
