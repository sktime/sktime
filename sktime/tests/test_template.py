import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from skltemplate import TemplateEstimator
from skltemplate import TemplateTransformer
from skltemplate import TemplateClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_template_estimator(data):
    est = TemplateEstimator()
    assert est.demo_param == 'demo_param'

    est.fit(*data)
    assert hasattr(est, 'is_fitted_')

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


def test_template_transformer_error(data):
    X, y = data
    trans = TemplateTransformer()
    trans.fit(X)
    with pytest.raises(ValueError, match="Shape of input is different"):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        trans.transform(X_diff_size)


def test_template_transformer(data):
    X, y = data
    trans = TemplateTransformer()
    assert trans.demo_param == 'demo'

    trans.fit(X)
    assert trans.n_features_ == X.shape[1]

    X_trans = trans.transform(X)
    assert_allclose(X_trans, np.sqrt(X))

    X_trans = trans.fit_transform(X)
    assert_allclose(X_trans, np.sqrt(X))


def test_template_classifier(data):
    X, y = data
    clf = TemplateClassifier()
    assert clf.demo_param == 'demo'

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
