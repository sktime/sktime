"""Test Experiments.

tests the experiments code works with all current valid classifiers. Loops through all
classifiers in the list in experiments, trains,  Builds all.
"""

__author__ = ["TonyBagnall"]


class TestStats:
    """Test performance statistics."""

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
}
