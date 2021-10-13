# -*- coding: utf-8 -*-
"""Working area to check changes dont break classifier building during refactor."""
from sktime.benchmarking.experiments import run_classification_experiment
from sktime.benchmarking.experiments import set_classifier
from sktime.datasets import load_unit_test

if __name__ == "__main__":
    trainX, trainY = load_unit_test(split="train", return_X_y=True)
    testX, testY = load_unit_test(split="test", return_X_y=True)
    classifier = "WEASEL"
    clf = set_classifier(classifier)
    clf.fit(trainX, trainY)
    y_pred = clf.predict(testX)
    y_pred_prob = clf.predict_proba(testX)
    print("Pred = ",y_pred[0])
    print("Proba = ",y_pred_prob[0])
    run_classification_experiment(trainX,trainY,testX,testY,clf,"C:/Temp/")
