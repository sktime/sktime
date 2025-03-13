"""Test function for TabTransformerRegressor"""

__author__ = ["Ankit-1204"]

import numpy as np
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
    seq_length = 10
    num_cat_class = [3, 2, 2]
    cat_idx = [1, 3, 4]
    n_dimensions = 1

    X, y = generate_seq(
        n_instances, n_features, seq_length, num_cat_class, cat_idx, n_dimensions
    )
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, train_size=0.7, random_state=100
    )

    model = TabTransformerRegressor(num_cat_class, cat_idx)
    model.fit(x_train, y_train)
    model.predict(x_test)


def generate_seq(
    n_instances, n_features, seq_length, num_cat_class, cat_idx, n_dimensions=1
):
    np.random.seed(100)
    X = np.zeros((n_instances, n_features, seq_length))
    for i, cat_col in enumerate(cat_idx):
        X[:, cat_col, :] = np.random.randint(
            0, num_cat_class[i], size=(n_instances, seq_length)
        )

    cont_idx = list(set(range(n_features)) - set(cat_idx))
    X[:, cont_idx, :] = np.random.randn(n_instances, len(cont_idx), seq_length)

    y = np.random.randn(n_instances, n_dimensions)

    return X, y
