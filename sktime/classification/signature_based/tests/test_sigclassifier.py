# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score
from sktime.datasets import load_gunpoint
from sktime.classification.signature_based import SignatureClassifier


def test_signatures_on_gunpoint():
    # Load data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)

    # Fit a simple sig classifier
    clf = SignatureClassifier(random_state=0)
    clf.fit(X_train, y_train)

    # Test and check accuracy
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    preds_test = clf.predict(X_test)
    accuracy = accuracy_score(preds_test, y_test)
    assert accuracy == 0.96
