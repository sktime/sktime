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


class TestStats:
    def __init__(self, chinatown_acc=0.0, multivariate=False, unequal_length=False,
                 missing_values=False):
        self.chinatown_acc = chinatown_acc
        self.multivariate = multivariate
        self.unequal_length = unequal_length
        self.missing_values = missing_values


# Map of classifier onto expected accuracy and expected capabilities
# If the algorithm is changed or capabilities are altered, then this needs to be
# verified and this file updated.
expected_capabilities = {
    "ProximityForest": TestStats(),
    "KNeighborsTimeSeriesClassifier": TestStats(),
    "ElasticEnsemble": TestStats(),
    "ShapeDTW": TestStats(),
    "BOSS": TestStats(chinatown_acc=0.9037900874635568),
    "ContractableBOSS": TestStats(chinatown_acc=0.9416909620991254),
    "TemporalDictionaryEnsemble": TestStats(chinatown_acc=0.9475218658892128),
    "WEASEL": TestStats(),
    "MUSE": TestStats(multivariate=True),
    "RandomIntervalSpectralForest": TestStats(chinatown_acc=0.9446064139941691),
    "TimeSeriesForest": TestStats(chinatown_acc=0.9708454810495627),
    "CanonicalIntervalForest": TestStats(chinatown_acc=0.9766763848396501),
    "ShapeletTransformClassifier": TestStats(chinatown_acc=0.0),
    # "ROCKET": TestStats(multivariate=True),
    "MrSEQLClassifier": TestStats(chinatown_acc=0.9737609329446064),
}


# Test that the classifiers listed in classification.base all
def test_classifiers_on_chinatown():
    path = os.path.join(os.path.dirname(__file__), "datasets/data")

    train_x, train_y = load_from_tsfile_to_dataframe(
        os.path.join(path, "Chinatown/Chinatown_TRAIN.ts")
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        os.path.join(path, "Chinatown/Chinatown_TEST.ts")
    )
    for name in range(0, len(classifier_list)):
        cls = set_classifier(name)
        # Test capabilities match expected
        assert cls._tags["multivariate"] == expected_capabilities[name].multivariate
        assert cls._tags["unequal_length"] == expected_capabilities[name].unequal_length
        assert cls._tags["missing_values"] == expected_capabilities[name].missing_values
        # Test observed accuracy matches expected accuracy
        # cls.fit(train_x, train_y)
        # preds = cls.predict(test_x)
        # ac = accuracy_score(test_y, preds)
        # assert abs(ac - expected_capabilities[name]) < 0.01


# def test_run_experiment_on_chinatown():
#    path = os.path.join(os.path.dirname(__file__), "datasets/data")#
#
#    train_x, train_y = load_from_tsfile_to_dataframe(
#        os.path.join(path, "ArrowHead/Chinatown_TRAIN.ts")
#    )
#    test_x, test_y = load_from_tsfile_to_dataframe(
#        os.path.join(path, "Chinatown/Chinatown_TEST.ts")
#    )

#    for name in range(0, len(classifier_list)):
#        cls = set_classifier(name)


# Test whether resampling is deteriministic. Resampling on fold 0 should give default
# Hard code test resampling on fold 1
# def test_stratified_resample():
#    path = os.path.join(os.path.dirname(__file__), "datasets/data")
#    train_x, train_y = load_from_tsfile_to_dataframe(
#        os.path.join(path, "Chinatown/Chinatown_TRAIN.ts")
#    )
