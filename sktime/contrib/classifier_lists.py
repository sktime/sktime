"""
contains lists of available classifiers and a method to create a default instance of each

"""

__author__ = "Anthony Bagnall"

__all__=[""]

from sktime.classification.dictionary_based import BOSSEnsemble, TemporalDictionaryEnsemble, MUSE, \
    WEASEL
from sktime.classification.distance_based import ProximityForest, ElasticEnsemble, \
    KNeighborsTimeSeriesClassifier
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier, MrSEQLClassifier, RocketClassifier

all_classifiers = [
    "BOSSEnsemble", "TemporalDictionaryEnsemble", "MUSE", "WEASEL",
    "ProximityForest", "ElasticEnsemble", "KNeighborsTimeSeriesClassifier",
#"ShapeDTW",
    "RandomIntervalSpectralForest",
    "TimeSeriesForest",
    "ShapeletTransformClassifier", "MrSEQLClassifier", "RocketClassifier"
]


def set_classifier(cls, resampleId):
    """
    Basic way of determining the classifier to build. To differentiate settings just and another elif. So, for example, if
    you wanted tuned TSF, you just pass TuneTSF and set up the tuning mechanism in the elif.
    This may well get superceded, it is just how e have always done it
    :param cls: String indicating which classifier you want
    :return: A classifier.

    """
    if cls.lower() == "boss":
        return BOSSEnsemble(random_state=resampleId)
    elif cls.lower() == "cboss":
        return BOSSEnsemble(
            random_state=resampleId, randomised_ensemble=True, max_ensemble_size=50
        )
    elif cls.lower() == "tde":
        return TemporalDictionaryEnsemble(random_state=resampleId)
    elif cls.lower() == "muse":
        return MUSE(random_state=resampleId)
    elif cls.lower() == "weasel":
        return WEASEL(random_state=resampleId)
    elif cls.lower() == "pf":
        return ProximityForest(random_state=resampleId)
    elif cls.lower() == "dtwcv":
        return KNeighborsTimeSeriesClassifier(metric="dtwcv")
    elif cls.lower() == "ee" or cls.lower() == "elasticensemble":
        return ElasticEnsemble()
#    elif cls.lower() == "shapedtw":
#        return ShapeDTW(random_state=resampleId)

    elif cls.lower() == "rise":
        return RandomIntervalSpectralForest(random_state=resampleId)
    elif cls.lower() == "tsf":
        return TimeSeriesForest(random_state=resampleId)
    elif cls.lower() == "tde":
        return TemporalDictionaryEnsemble(random_state=resampleId)
    elif cls.lower() == "st":
        return ShapeletTransformClassifier(time_contract_in_mins=1500)
    else:
        raise Exception("UNKNOWN CLASSIFIER")