# -*- coding: utf-8 -*-
""" test_experiments.py, tests the experiments code works with all current valid
classifiers.

Loops through all classifiers in the list in experiments, trains,  Builds all
"""
import os
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.classification.base import classifier_list
from sktime.contrib.classification_experiments import set_classifier
from sklearn.metrics import accuracy_score
import sktime.datasets.base as sktime


class TestStats:
    def __init__(
        self,
        unit_test_acc=0.0,
        multivariate=False,
        unequal_length=False,
        missing_values=False,
    ):
        self.chinatown_acc = unit_test_acc
        self.multivariate = multivariate
        self.unequal_length = unequal_length
        self.missing_values = missing_values


# Map of classifier onto expected accuracy and expected capabilities
# If the algorithm is changed or capabilities are altered, then this needs to be
# verified and this file updated.
expected_capabilities = {
    "ProximityForest": TestStats(unit_test_acc=0.8636363636363636),
    "KNeighborsTimeSeriesClassifier": TestStats(unit_test_acc=0.8636363636363636),
    "ElasticEnsemble": TestStats(unit_test_acc=0.8636363636363636),
    "ShapeDTW": TestStats(unit_test_acc=0.8636363636363636),
    "BOSS": TestStats(unit_test_acc=0.7727272727272727),
    "ContractableBOSS": TestStats(unit_test_acc=0.8636363636363636),
    "TemporalDictionaryEnsemble": TestStats(unit_test_acc=0.8636363636363636),
    "WEASEL": TestStats(unit_test_acc=0.7272727272727273),
    "MUSE": TestStats(unit_test_acc=0.7272727272727273, multivariate=True),
    "RandomIntervalSpectralForest": TestStats(unit_test_acc=0.8636363636363636),
    "TimeSeriesForest": TestStats(unit_test_acc=0.9090909090909091),
    "CanonicalIntervalForest": TestStats(unit_test_acc=0.8636363636363636),
    "ShapeletTransformClassifier": TestStats(unit_test_acc=0.8636363636363636),
    "ROCKET": TestStats(unit_test_acc=0.9090909090909091, multivariate=True),
    "MrSEQLClassifier": TestStats(unit_test_acc=0.8636363636363636),
}


# Test that the classifiers listed in classification.base all
def test_classifiers_on_default_problem():
    path = os.path.join(sktime.MODULE, "datasets/data")
    # train_x, train_y = load_from_tsfile_to_dataframe(
    #    os.path.join(path, "UnitTest/UnitTest_TRAIN.ts")
    # )
    # test_x, test_y = load_from_tsfile_to_dataframe(
    #    os.path.join(path, "UnitTest/UnitTest_TEST.ts")
    # )
    # for name in range(0, len(classifier_list)):
    #    cls = set_classifier(name)
    # Test capabilities match expected
    # assert (
    #         cls.capabilities["multivariate"] ==
    #         expected_capabilities[name].multivariate
    # )
    # assert (
    #         cls.capabilities["unequal_length"] ==
    #         expected_capabilities[name].unequal_length
    # )
    # assert (
    #         cls.capabilities["missing_values"] ==
    #         expected_capabilities[name].missing_values
    # )
    # # Test observed accuracy matches expected accuracy
    # cls.fit(train_x, train_y)
    # preds = cls.predict(test_x)
    # ac = accuracy_score(test_y, preds)
    # assert abs(ac - expected_capabilities[name]) < 0.01


# def test_run_experiment_on_chinatown():
#    path = os.path.join(os.path.dirname(__file__), "datasets/data")#
#
#    train_x, train_y = load_from_tsfile_to_dataframe(
#        os.path.join(path, "ArrowHead/UnitTest_TRAIN.ts")
#    )
#    test_x, test_y = load_from_tsfile_to_dataframe(
#        os.path.join(path, "UnitTest/UnitTest_TEST.ts")
#    )

#    for name in range(0, len(classifier_list)):
#        cls = set_classifier(name)


# Test whether resampling is deteriministic. Resampling on fold 0 should give default
# Hard code test resampling on fold 1
# def test_stratified_resample():
#    path = os.path.join(os.path.dirname(__file__), "datasets/data")
#    train_x, train_y = load_from_tsfile_to_dataframe(
#        os.path.join(path, "UnitTest/UnitTest_TRAIN.ts")
#    )
