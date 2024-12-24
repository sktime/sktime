import pytest
from sktime.clustering.base import BaseClusterer
from sktime.registry import all_estimators

class TestAllClusterers:
    @pytest.mark.parametrize("Estimator", all_estimators(estimator_type="clusterer"))
    def test_clusterer(self, Estimator):
        assert issubclass(Estimator, BaseClusterer)
        # Additional tests for clusterers can be added here
