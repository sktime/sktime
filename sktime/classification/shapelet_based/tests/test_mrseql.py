import numpy as np
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_gunpoint


def test_mrseql_on_gunpoint():
    # load training data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)

    sax_clf = MrSEQLClassifier(seql_mode='fs', symrep=['sax'])
    sfa_clf = MrSEQLClassifier(seql_mode='fs', symrep=['sfa'])
    ss_clf = MrSEQLClassifier(seql_mode='fs', symrep=['sax', 'sfa'])

    # fit training data
    sax_clf.fit(X_train, y_train)
    sfa_clf.fit(X_train, y_train)
    ss_clf.fit(X_train, y_train)

    # prediction
    sax_predicted = sax_clf.predict(X_test)
    sfa_predicted = sfa_clf.predict(X_test)
    ss_predicted = ss_clf.predict(X_test)

    # test feature space dimension
    # the multi-domain classifier (ss_clf) should produce as many features
    # as the others (sax_clf and sfa_clf) combine
    np.testing.assert_equal(ss_clf.ots_clf.coef_.shape[1],
                            sfa_clf.ots_clf.coef_.shape[1] +
                            sax_clf.ots_clf.coef_.shape[1])

    # test number of correct predictions
    np.testing.assert_equal((sax_predicted == y_test).sum(), 148)
    np.testing.assert_equal((sfa_predicted == y_test).sum(), 150)
    np.testing.assert_equal((ss_predicted == y_test).sum(), 150)
