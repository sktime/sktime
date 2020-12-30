# -*- coding: utf-8 -*-
""" test_experiments.py, tests the experiments code works with all current valid
classifiers.

Loops through all classifiers in the list in experiments, trains,  Builds all
"""
import os
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.classification.base import classifier_list
from sktime.contrib.experiments import set_classifier
from sklearn.metrics import accuracy_score

# Map of classifier onto expected accuracy. This will need adjusting for any new
expected_accuracy = {
    "ProximityForest": 0.0,
    "KNeighborsTimeSeriesClassifier": 0.0,
    "ElasticEnsemble": 0.0,
    "ShapeDTW": 0.0,
    "BOSS": 0.9037900874635568,
    "ContractableBOSS": 0.9416909620991254,
    "TemporalDictionaryEnsemble": 0.9475218658892128,
    "WEASEL": 0.0,
    "MUSE": 0.0,
    "RandomIntervalSpectralForest": 0.9446064139941691,
    "TimeSeriesForest": 0.0,
    "CanonicalIntervalForest": 0.9766763848396501,
    "ShapeletTransformClassifier": 0.0,
    "ROCKET": 0.0,
    "MrSEQLClassifier": 0.9737609329446064,
}


# Test whether resampling is deteriministic. Resampling on fold 0 should give default
# Hard code test resampling on fold 1
def test_stratified_resample():
    path = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
    train_x, train_y = load_from_tsfile_to_dataframe(
        os.path.join(path, "Chinatown/Chinatown_TRAIN.ts")
    )


# Test that the classifiers listed in classification.base all
def test_set_classifier_on_chinatown():
    path = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")

    train_x, train_y = load_from_tsfile_to_dataframe(
        os.path.join(path, "Chinatown/Chinatown_TRAIN.ts")
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        os.path.join(path, "Chinatown/Chinatown_TEST.ts")
    )
    for name in range(0, len(classifier_list)):
        cls = set_classifier(name)
        cls.fit(train_x, train_y)
        preds = cls.predict(test_x)
        ac = accuracy_score(test_Y, preds)
        assert abs(ac - expected_accuracy[name]) < 0.01


def test_run_experiment_on_chinatown():
    path = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")

    train_x, train_y = load_from_tsfile_to_dataframe(
        os.path.join(path, "ArrowHead/Chinatown_TRAIN.ts")
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        os.path.join(path, "Chinatown/Chinatown_TEST.ts")
    )
    for name in range(0, len(classifier_list)):
        cls = set_classifier(name)

