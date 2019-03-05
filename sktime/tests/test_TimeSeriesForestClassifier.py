from ..classifiers.ensemble import TimeSeriesForestClassifier
from ..utils.testing import generate_df_from_array
import pytest
import pandas as pd
import numpy as np

N_ITER = 10

n = 20
d = 1
m = 20
n_classes = 2

X = generate_df_from_array(np.random.normal(size=m), n_rows=n, n_cols=d)
y = pd.Series(np.random.choice(np.arange(n_classes) + 1, size=n))


# Check if random state always gives same results
def test_random_state():
    random_state = 1234
    clf = TimeSeriesForestClassifier(n_estimators=2,
                                     random_state=random_state)
    clf.fit(X, y)
    first_pred = clf.predict_proba(X)
    for _ in range(N_ITER):
        clf = TimeSeriesForestClassifier(n_estimators=2,
                                         random_state=random_state)
        clf.fit(X, y)
        y_pred = clf.predict_proba(X)
        np.testing.assert_array_equal(first_pred, y_pred)


def test_predict_proba():
    clf = TimeSeriesForestClassifier(n_estimators=2)
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], n_classes)
    np.testing.assert_array_equal(np.ones(n), np.sum(proba, axis=1))

