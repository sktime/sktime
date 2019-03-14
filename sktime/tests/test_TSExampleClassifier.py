from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sklearn.utils.testing import assert_array_equal

from sktime.classifiers.example_classifiers import TSExampleClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sktime.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sktime.datasets import load_gunpoint

Xsf_train, y_train = load_gunpoint(return_X_y=True)
Xdf_train = pd.DataFrame({'ts': Xsf_train, 'ts_copy': Xsf_train})
Xsf_test, y_test = load_gunpoint("TEST", return_X_y=True)
Xdf_test = pd.DataFrame({'ts': Xsf_test, 'ts_copy': Xsf_test})

def test_xdataframe_TSExampleClassifier():
    X = Xdf_train
    y = y_train
    model = TSExampleClassifier(func=np.mean, columns=X.columns, estimator=RandomForestClassifier(random_state=123, n_estimators=10))
    model.fit(X, y)
    assert_array_equal(model.predict(Xdf_test), np.ones(y_test.shape[0]) * 2)

def test_set_get_param():
    X = Xdf_train
    y = y_train
    model = TSExampleClassifier(func=np.mean, columns=X.columns, estimator=RandomForestClassifier(random_state=123, n_estimators=10))
    model.set_params(estimator__random_state=42)
    assert model.get_params()['estimator__random_state'] == 42

def test_grid_search_cv():
    X = Xdf_train
    y = y_train
    model = TSExampleClassifier(func=np.mean,
                                columns=X.columns,
                                estimator=LogisticRegression(fit_intercept=True,
                                                             solver='lbfgs'))
    model.fit(X, y)
    expected = model.predict(X)

    # give (deep) parameter tuning details
    parameters = {'estimator__fit_intercept': (True, False)}
    # as we are not using a mixin, we need an external scorer
    external_scorer = make_scorer(accuracy_score)
    # fit and predict GridSearchCV
    clf = GridSearchCV(model, parameters, scoring=external_scorer, cv=5)
    clf.fit(X, y)
    got = clf.predict(X)
    assert_array_equal(expected, got)

def test_grid_search_cv_default_scorer():
    X = Xdf_train
    y = y_train
    model = TSExampleClassifier(func=np.mean,
                                columns=X.columns,
                                estimator=LogisticRegression(fit_intercept=True,
                                                             solver='lbfgs'))
    model.fit(X, y)
    expected = model.predict(X)

    # give (deep) parameter tuning details
    parameters = {'estimator__fit_intercept': (True, False)}
    # fit and predict GridSearchCV without external scorer
    clf = GridSearchCV(model, parameters, cv=5)
    clf.fit(X, y)
    got = clf.predict(X)
    assert_array_equal(expected, got)
