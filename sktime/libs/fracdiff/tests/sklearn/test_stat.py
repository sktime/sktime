#import numpy as np
#import pytest
#
#from fracdiff.sklearn.stat import StatTester
#
#
#class TestStat:
#    """
#    Test `StatTester`.
#    """
#
#    def _make_stationary(self, seed, n_samples):
#        np.random.seed(seed)
#        return np.random.randn(n_samples)
#
#    def _make_nonstationary(self, seed, n_samples):
#        np.random.seed(seed)
#        return np.random.randn(n_samples).cumsum()
#
#    @pytest.mark.parametrize("seed", [42])
#    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
#    def test_stationary(self, seed, n_samples):
#        X = self._make_stationary(seed, n_samples)
#
#        assert StatTester().pvalue(X) < 0.1
#        assert StatTester().is_stat(X)
#
#    @pytest.mark.parametrize("seed", [42])
#    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
#    def test_nonstationary(self, seed, n_samples):
#        X = self._make_nonstationary(seed, n_samples)
#
#        assert StatTester().pvalue(X) > 0.1
#        assert not StatTester().is_stat(X)
#
#    def test_method_valueerror(self):
#        tester = StatTester(method="nonexistent")
#
#        with pytest.raises(ValueError):
#            _ = tester.null_hypothesis
#        with pytest.raises(ValueError):
#            _ = tester.pvalue(np.ones(100))
#
