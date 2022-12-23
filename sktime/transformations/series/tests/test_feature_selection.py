# -*- coding: utf-8 -*-
"""Test FeatureSelection transformer."""

__author__ = ["aiwalter"]
__all__ = []

import math

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.tree import DecisionTreeRegressor

from sktime.datasets import load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.feature_selection import FeatureSelection

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=3)


@pytest.mark.parametrize(
    "method", ["feature-importances", "random", "columns", "none", "all"]
)
@pytest.mark.parametrize("n_columns", [None, 2])
@pytest.mark.parametrize("random_state", [None, 1])
def test_feature_selection(method, n_columns, random_state):
    columns = ["GNP", "UNEMP"] if method == "columns" else None
    transformer = FeatureSelection(
        method=method, columns=columns, n_columns=n_columns, random_state=random_state
    )
    transformer.fit(X=X_train, y=y_train)
    X_hat = transformer.transform(X=X_test, y=y_test)
    if method != "none":
        assert isinstance(X_hat, pd.DataFrame)
    else:
        assert X_hat is None

    if method == "feature-importances":
        if n_columns is None:
            n_columns = int(math.ceil(X_train.shape[1] / 2))
        else:
            assert X_hat.shape[1] == n_columns
        assert isinstance(transformer.feature_importances_, dict)
        assert len(transformer.feature_importances_) == X_train.shape[1]
        assert isinstance(transformer.feature_importances_, dict)
        # test custom regressor
        transformer_f1 = FeatureSelection(
            method=method, regressor=DecisionTreeRegressor()
        )
        transformer_f1.fit(X=X_train, y=y_train)
        _ = transformer_f1.transform(X=X_test, y=y_test)
        transformer_f2 = FeatureSelection(method=method)
        transformer_f2.fit(X=X_train, y=y_train)
        _ = transformer_f2.transform(X=X_test, y=y_test)

        assert (
            transformer_f1.feature_importances_ != transformer_f2.feature_importances_
        )
    if method == "random":
        if n_columns is None:
            n_columns = int(math.ceil(X_train.shape[1] / 2))
        else:
            assert X_hat.shape[1] == n_columns
            # test random state
            transformer_rand1 = FeatureSelection(method=method, random_state=None)
            transformer_rand1.fit(X_train)
            X_hat_rand1 = transformer_rand1.transform(X_test)

            transformer_rand2 = FeatureSelection(method=method, random_state=3)
            transformer_rand2.fit(X_train)
            X_hat_rand2 = transformer_rand2.transform(X_test)

            with pytest.raises(AssertionError):
                assert_frame_equal(X_hat_rand1, X_hat_rand2)
    if method == "columns":
        if columns is None:
            assert X_hat.shape[1] == X_train.shape[1]
        else:
            assert X_hat.shape[1] == len(columns)
            for c in columns:
                assert c in X_hat.columns
