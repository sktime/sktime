"""Test function for TabTransformerRegressor"""

__author__ = ["Ankit-1204"]

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sktime.regression.deep_learning.tab_transformer import TabTransformerRegressor
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TabTransformerRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tab_tranformer():
    n_instances = 10
    n_features = 6
    seq_length = 9
    num_cat_class = [3, 2, 2]
    cat_idx = [1, 3, 4]
    n_dimensions = 1

    X, y = generate_seq(
        n_instances, n_features, seq_length, num_cat_class, cat_idx, n_dimensions
    )
    instances = X.index.get_level_values("instance").unique()
    train_instances, test_instances = train_test_split(
        instances, test_size=0.3, train_size=0.7, random_state=100
    )
    x_train = X.loc[train_instances]
    x_test = X.loc[test_instances]
    y_train = y[train_instances]

    model = TabTransformerRegressor(num_cat_class, cat_idx)
    model.fit(x_train, y_train)
    model.predict(x_test)


def generate_seq(
    n_instances, n_features, seq_length, num_cat_class, cat_idx, n_dimensions=1
):
    np.random.seed(100)
    index = pd.MultiIndex.from_product(
        [range(n_instances), range(seq_length)], names=["instance", "time"]
    )
    X = np.zeros((n_instances * seq_length, n_features))
    for i, cat_col in enumerate(cat_idx):
        cat_data = np.random.randint(
            0, num_cat_class[i], size=(n_instances, seq_length)
        ).flatten()
        X[:, cat_col] = cat_data

    cont_idx = list(set(range(n_features)) - set(cat_idx))
    X[:, cont_idx] = np.random.randn(n_instances * seq_length, len(cont_idx))
    X = pd.DataFrame(X, index=index, columns=[f"feat_{i}" for i in range(n_features)])
    for col in cat_idx:
        X[f"feat_{col}"] = X[f"feat_{col}"].astype("category")

    y = np.random.randn(n_instances, n_dimensions)
    print(f"X : {X}")
    print(f"y : {y}")
    return X, y
