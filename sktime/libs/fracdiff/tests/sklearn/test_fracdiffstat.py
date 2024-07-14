#import numpy as np
#import pandas as pd
#import pytest
#from numpy.testing import assert_allclose
#
#from fracdiff.sklearn import Fracdiff
#from fracdiff.sklearn import FracdiffStat
#from fracdiff.sklearn.stat import StatTester
#
#
#class TestFracdiffStat:
#    @staticmethod
#    def _is_stat(x):
#        return StatTester().is_stat(x)
#
#    @pytest.mark.parametrize("window", [10])
#    @pytest.mark.parametrize("mode", ["same", "valid"])
#    @pytest.mark.parametrize("precision", [0.01])
#    @pytest.mark.parametrize("n_jobs", [None, -1])
#    def test_order(self, window, mode, precision, n_jobs):
#        np.random.seed(42)
#        X = np.random.randn(100, 2).cumsum(0)
#
#        fs = FracdiffStat(mode=mode, window=window, precision=precision, n_jobs=n_jobs)
#        fs.fit(X)
#
#        X_st = fs.transform(X)
#        X_ns = np.empty_like(X_st[:, :0])
#
#        for i in range(X.shape[1]):
#            f = Fracdiff(fs.d_[i] - precision, mode=mode, window=window)
#            X_ns = np.concatenate((X_ns, f.fit_transform(X[:, [i]])), 1)
#
#        for i in range(X.shape[1]):
#            assert self._is_stat(X_st[:, i])
#            assert not self._is_stat(X_ns[:, i])
#
#    @pytest.mark.parametrize("window", [10])
#    def test_lower_is_stat(self, window):
#        """
#        Test if `StationarityFracdiff.fit` returns `lower`
#        if `lower`th differenciation is already stationary.
#        """
#        np.random.seed(42)
#        X = np.random.randn(100, 1)
#
#        f = FracdiffStat(window=window, lower=0.0).fit(X)
#
#        assert f.d_[0] == 0.0
#
#    @pytest.mark.parametrize("window", [10])
#    def test_upper_is_not_stat(self, window):
#        """
#        Test if `StationarityFracdiff.fit` returns `np.nan`
#        if `upper`th differenciation is still non-stationary.
#        """
#        np.random.seed(42)
#        X = np.random.randn(100, 1).cumsum(0)
#
#        f = FracdiffStat(window=window, upper=0.0, lower=-1.0)
#        with pytest.raises(RuntimeWarning):
#            f.fit(X)
#
#        assert np.isnan(f.d_[0])
#
#    @pytest.mark.parametrize("window", [10])
#    @pytest.mark.parametrize("mode", ["same", "valid"])
#    @pytest.mark.parametrize("precision", [0.01])
#    @pytest.mark.parametrize("n_jobs", [None, -1])
#    def test_transform(self, window, mode, precision, n_jobs):
#        """
#        Test if `FracdiffStat.transform` works
#        for array with n_features > 1.
#        """
#        np.random.seed(42)
#        X = np.random.randn(100, 2).cumsum(0)
#
#        fs = FracdiffStat(
#            window=window, mode=mode, precision=precision, n_jobs=n_jobs
#        ).fit(X)
#        out = fs.transform(X)
#
#        exp = np.empty_like(out[:, :0])
#        for i in range(X.shape[1]):
#            f = Fracdiff(fs.d_[i], mode=mode, window=window)
#            exp = np.concatenate((exp, f.fit_transform(X[:, [i]])), 1)
#
#        assert_allclose(out, exp)
#
#    def test_transform_pandas_dataframe(self):
#        np.random.seed(42)
#
#        X_np = np.random.randn(100, 2).cumsum(0)
#        X_pd = pd.DataFrame(X_np).copy()
#
#        output_np = FracdiffStat().fit_transform(X_np)
#        output_pd = FracdiffStat().fit_transform(X_pd)
#
#        assert_allclose(output_np, output_pd)
#
#    def test_import_deprecation(self):
#        with pytest.raises(DeprecationWarning):
#            from fracdiff import FracdiffStat
#
#            _ = FracdiffStat()
#
#    def test_full_deprecated(self):
#        np.random.seed(42)
#        X = np.random.randn(10, 10)
#        with pytest.raises(DeprecationWarning):
#            _ = Fracdiff(0.5, mode="full").fit_transform(X)
#
