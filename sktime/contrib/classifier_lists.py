"""
contains lists of available classifiers and a method to create a default instance of each

"""

__author__ = "Anthony Bagnall"

__all__=[""]

from sktime.classification.dictionary_based import BOSSEnsemble, TemporalDictionaryEnsemble, MUSE, \
    WEASEL
from sktime.classification.distance_based import ProximityForest, KNeighborsTimeSeriesClassifier, \
    ElasticEnsemble
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier, MrSEQLClassifier

all_classifiers=[
    "BOSSEnsemble", "TemporalDictionaryEnsemble", "MUSE", "WEASEL", #dictionary based
    "ProximityForest", "ElasticEnsemble", "ShapeDTW", "KNeighborsTimeSeriesClassifier",
    "RandomIntervalSpectralForest",
    "TimeSeriesForest",
    "ShapeletTransformClassifier", "MrSEQLClassifier"
]

def set_classifier(cls, resampleId):
    """
    Basic way of determining the classifier to build. To differentiate settings just and another elif. So, for example, if
    you wanted tuned TSF, you just pass TuneTSF and set up the tuning mechanism in the elif.
    This may well get superceded, it is just how e have always done it
    :param cls: String indicating which classifier you want
    :return: A classifier.

    """
    if cls.lower() == "pf":
        return ProximityForest(random_state=resampleId)
    elif cls.lower() == "pt":
        return ProximityTree(random_state=resampleId)
    elif cls.lower() == "ps":
        return ProximityStump(random_state=resampleId)
    elif cls.lower() == "rise":
        return RandomIntervalSpectralForest(random_state=resampleId)
    elif cls.lower() == "tsf":
        return TimeSeriesForest(random_state=resampleId)
    elif cls.lower() == "boss":
        return BOSSEnsemble(random_state=resampleId)
    elif cls.lower() == "cboss":
        return BOSSEnsemble(
            random_state=resampleId, randomised_ensemble=True, max_ensemble_size=50
        )
    elif cls.lower() == "tde":
        return TemporalDictionaryEnsemble(random_state=resampleId)
    elif cls.lower() == "st":
        return ShapeletTransformClassifier(time_contract_in_mins=1500)
    elif cls.lower() == "dtwcv":
        return KNeighborsTimeSeriesClassifier(metric="dtwcv")
    elif cls.lower() == "ee" or cls.lower() == "elasticensemble":
        return ElasticEnsemble()
    # elif cls.lower() == "tsfcomposite":
    #     # It defaults to TS
    #     return TimeSeriesForestClassifier()
    # elif cls.lower() == "risecomposite":
    #     steps = [
    #         ("segment", RandomIntervalSegmenter(n_intervals=1, min_length=5)),
    #         (
    #             "transform",
    #             FeatureUnion(
    #                 [
    #                     (
    #                         "acf",
    #                         make_row_transformer(
    #                             FunctionTransformer(func=acf_coefs, validate=False)
    #                         ),
    #                     ),
    #                     (
    #                         "ps",
    #                         make_row_transformer(
    #                             FunctionTransformer(func=powerspectrum, validate=False)
    #                         ),
    #                     ),
    #                 ]
    #             ),
    #         ),
    #         ("tabularise", Tabularizer()),
    #         ("clf", DecisionTreeClassifier()),
    #     ]
    #     base_estimator = Pipeline(steps)
    #     return ensemble.TimeSeriesForestClassifier(
    #         estimator=base_estimator, n_estimators=100
    #     )
    else:
        raise Exception("UNKNOWN CLASSIFIER")