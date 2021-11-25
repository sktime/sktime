# -*- coding: utf-8 -*-
""" catch22 features
A transformer for the catch22 features
"""

__author__ = "Matthew Middlehurst"
__all__ = ["Catch22"]

import math

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sktime.utils.validation.panel import check_X


class Catch22(_PanelToTabularTransformer):
    """Canonical Time-series Characteristics (catch22)

    @article{lubba2019catch22,
        title={catch22: CAnonical Time-series CHaracteristics},
        author={Lubba, Carl H and Sethi, Sarab S and Knaute, Philip and
                Schultz, Simon R and Fulcher, Ben D and Jones, Nick S},
        journal={Data Mining and Knowledge Discovery},
        volume={33},
        number={6},
        pages={1821--1852},
        year={2019},
        publisher={Springer}
    }

    Overview: Input n series with d dimensions of length m
    Transforms series into the 22 catch22 features extracted from the hctsa
    toolbox.

    Fulcher, B. D., & Jones, N. S. (2017). hctsa: A computational framework
    for automated time-series phenotyping using massive feature extraction.
    Cell systems, 5(5), 527-531.

    Fulcher, B. D., Little, M. A., & Jones, N. S. (2013). Highly comparative
    time-series analysis: the empirical structure of time series and their
    methods. Journal of the Royal Society Interface, 10(83), 20130048.

    Original catch22 package implementations:
    https://github.com/chlubba/catch22

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/transformers/Catch22.java

    """

    def __init__(
        self,
        outlier_norm=False,
        n_jobs=1,
    ):
        self.outlier_norm = outlier_norm

        self.n_jobs = n_jobs

        super(Catch22, self).__init__()

    def transform(self, X, y=None):
        """transforms data into the catch22 features

        Parameters
        ----------
        X : pandas DataFrame or 3d numpy array, input time series
        y : array_like, target values (optional, ignored)

        Returns
        -------
        Pandas dataframe containing 22 features for each input series
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)
        n_instances = X.shape[0]
        X = np.reshape(X, (n_instances, -1))

        c22_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform_case)(
                X[i],
            )
            for i in range(n_instances)
        )

        return pd.DataFrame(c22_list)

    def _transform_case(self, series):
        outlier_series = series
        if self.outlier_norm:
            std = np.std(outlier_series)
            if std > 0:
                outlier_series = (outlier_series - np.mean(outlier_series)) / std

        smin = np.min(series)
        smax = np.max(series)
        smean = np.mean(series)

        nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
        fft = np.fft.fft(series - smean, n=nfft)
        ac = _autocorr(series, fft)
        acfz = _ac_first_zero(ac)

        c22 = np.zeros(22)
        c22[0] = Catch22.DN_HistogramMode_5(series, smin, smax)
        c22[1] = Catch22.DN_HistogramMode_10(series, smin, smax)
        c22[2] = Catch22.SB_BinaryStats_diff_longstretch0(series, smean)
        c22[3] = Catch22.DN_OutlierInclude_p_001_mdrmd(outlier_series)
        c22[4] = Catch22.DN_OutlierInclude_n_001_mdrmd(outlier_series)
        c22[5] = Catch22.CO_f1ecac(ac)
        c22[6] = Catch22.CO_FirstMin_ac(ac)
        c22[7] = Catch22.SP_Summaries_welch_rect_area_5_1(series, fft)
        c22[8] = Catch22.SP_Summaries_welch_rect_centroid(series, fft)
        c22[9] = Catch22.FC_LocalSimple_mean3_stderr(series)
        c22[10] = Catch22.CO_trev_1_num(series)
        c22[11] = Catch22.CO_HistogramAMI_even_2_5(series, smin, smax)
        c22[12] = Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(ac)
        c22[13] = Catch22.MD_hrv_classic_pnn40(series)
        c22[14] = Catch22.SB_BinaryStats_mean_longstretch1(series)
        c22[15] = Catch22.SB_MotifThree_quantile_hh(series)
        c22[16] = Catch22.FC_LocalSimple_mean1_tauresrat(series, acfz)
        c22[17] = Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(series, acfz)
        c22[18] = Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(series)
        c22[19] = Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(series)
        c22[20] = Catch22.SB_TransitionMatrix_3ac_sumdiagcov(series, acfz)
        c22[21] = Catch22.PD_PeriodicityWang_th0_01(series)

        return c22

    def _transform_single_feature(self, X, feature):
        """transforms data into a specified catch22 feature

        Parameters
        ----------
        X : pandas DataFrame, input time series
        feature : int, catch22 feature id or String, catch22 feature
                  name.

        Returns
        -------
        Numpy array containing a catch22 feature for each input series
        """
        if isinstance(feature, (int, np.integer)) or isinstance(
            feature, (float, np.float)
        ):
            if feature > 21 or feature < 0:
                raise ValueError("Invalid catch22 feature ID")
        elif isinstance(feature, str):
            if feature in feature_names:
                feature = feature_names.index(feature)
            else:
                raise ValueError("Invalid catch22 feature name")
        else:
            raise ValueError("catch22 feature name or ID required")

        if isinstance(X, pd.DataFrame):
            X = from_nested_to_2d_array(X, return_numpy=True)

        n_instances = X.shape[0]
        X = np.reshape(X, (n_instances, -1))

        c22_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform_case_single)(
                X[i],
                feature,
            )
            for i in range(n_instances)
        )

        return np.asarray(c22_list)

    def _transform_case_single(self, series, feature):
        args = [series]

        if feature == 0 or feature == 1 or feature == 11:
            smin = np.min(series)
            smax = np.max(series)
            args = [series, smin, smax]
        elif feature == 2:
            smean = np.mean(series)
            args = [series, smean]
        elif feature == 3 or feature == 4:
            if self.outlier_norm:
                std = np.std(series)
                if std > 0:
                    series = (series - np.mean(series)) / std
            args = [series]
        elif feature == 7 or feature == 8:
            smean = np.mean(series)
            nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
            fft = np.fft.fft(series - smean, n=nfft)
            args = [series, fft]
        elif feature == 5 or feature == 6 or feature == 12:
            smean = np.mean(series)
            nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
            fft = np.fft.fft(series - smean, n=nfft)
            ac = _autocorr(series, fft)
            args = [ac]
        elif feature == 16 or feature == 17 or feature == 20:
            smean = np.mean(series)
            nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
            fft = np.fft.fft(series - smean, n=nfft)
            ac = _autocorr(series, fft)
            acfz = _ac_first_zero(ac)
            args = [series, acfz]

        return features[feature](*args)

    @staticmethod
    def DN_HistogramMode_5(X, smin, smax):
        # Mode of z-scored distribution (5-bin histogram)
        return _histogram_mode(X, 5, smin, smax)

    @staticmethod
    def DN_HistogramMode_10(X, smin, smax):
        # Mode of z-scored distribution (10-bin histogram)
        return _histogram_mode(X, 10, smin, smax)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def SB_BinaryStats_diff_longstretch0(X, smean):
        # Longest period of consecutive values above the mean
        mean_binary = np.zeros(len(X))
        for i in range(len(X)):
            if X[i] - smean > 0:
                mean_binary[i] = 1

        return _long_stretch(mean_binary, 1)

    @staticmethod
    def DN_OutlierInclude_p_001_mdrmd(X):
        # Time intervals between successive extreme events above the mean
        return _outlier_include(X)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def DN_OutlierInclude_n_001_mdrmd(X):
        # Time intervals between successive extreme events below the mean
        return _outlier_include(-X)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def CO_f1ecac(X_ac):
        # First 1/e crossing of autocorrelation function
        threshold = 0.36787944117144233  # 1 / np.exp(1)
        for i in range(1, len(X_ac)):
            if (X_ac[i - 1] - threshold) * (X_ac[i] - threshold) < 0:
                return i
        return len(X_ac)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def CO_FirstMin_ac(X_ac):
        # First minimum of autocorrelation function
        for i in range(1, len(X_ac) - 1):
            if X_ac[i] < X_ac[i - 1] and X_ac[i] < X_ac[i + 1]:
                return i
        return len(X_ac)

    @staticmethod
    def SP_Summaries_welch_rect_area_5_1(X, X_fft):
        # Total power in lowest fifth of frequencies in the Fourier power spectrum
        return _summaries_welch_rect(X, False, X_fft)

    @staticmethod
    def SP_Summaries_welch_rect_centroid(X, X_fft):
        # Centroid of the Fourier power spectrum
        return _summaries_welch_rect(X, True, X_fft)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def FC_LocalSimple_mean3_stderr(X):
        # Mean error from a rolling 3-sample mean forecasting
        if len(X) - 3 < 3:
            return 0
        res = _local_simple_mean(X, 3)
        return np.std(res)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def CO_trev_1_num(X):
        # Time-reversibility statistic, ((x_t+1 − x_t)^3)_t
        y = np.zeros(len(X) - 1)
        for i in range(len(y)):
            y[i] = np.power(X[i + 1] - X[i], 3)
        return np.mean(y)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def CO_HistogramAMI_even_2_5(X, smin, smax):
        # Automutual information, m = 2, τ = 5
        new_min = smin - 0.1
        new_max = smax + 0.1
        bin_width = (new_max - new_min) / 5

        histogram = np.zeros((5, 5))
        sumx = np.zeros(5)
        sumy = np.zeros(5)
        v = 1.0 / (len(X) - 2)
        for i in range(len(X) - 2):
            idx1 = int((X[i] - new_min) / bin_width)
            idx2 = int((X[i + 2] - new_min) / bin_width)

            histogram[idx1][idx2] += v
            sumx[idx1] += v
            sumy[idx2] += v

        nsum = 0
        for i in range(5):
            for n in range(5):
                if histogram[i][n] > 0:
                    nsum += histogram[i][n] * np.log(
                        histogram[i][n] / sumx[i] / sumy[n]
                    )

        return nsum

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def IN_AutoMutualInfoStats_40_gaussian_fmmi(X_ac):
        # First minimum of the automutual information function
        tau = int(min(40, np.ceil(len(X_ac) / 2)))

        diffs = np.zeros(tau - 1)
        prev = -0.5 * np.log(1 - np.power(X_ac[1], 2))
        for i in range(len(diffs)):
            corr = -0.5 * np.log(1 - np.power(X_ac[i + 2], 2))
            diffs[i] = corr - prev
            prev = corr

        for i in range(len(diffs) - 1):
            if diffs[i] * diffs[i + 1] < 0 and diffs[i] < 0:
                return i + 1

        return tau

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def MD_hrv_classic_pnn40(X):
        # Proportion of successive differences exceeding 0.04σ (Mietus 2002)
        diffs = np.zeros(len(X) - 1)
        for i in range(len(diffs)):
            diffs[i] = np.abs(X[i + 1] - X[i]) * 1000

        nsum = 0
        for diff in diffs:
            if diff > 40:
                nsum += 1

        return nsum / len(diffs)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def SB_BinaryStats_mean_longstretch1(X):
        # Longest period of successive incremental decreases
        diff_binary = np.zeros(len(X) - 1)
        for i in range(len(diff_binary)):
            if X[i + 1] - X[i] >= 0:
                diff_binary[i] = 1

        return _long_stretch(diff_binary, 0)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def SB_MotifThree_quantile_hh(X):
        # Shannon entropy of two successive letters in equiprobable 3-letter
        # symbolization
        indicies = np.argsort(X)
        bins = np.zeros(len(X))
        q1 = int(len(X) / 3)
        q2 = q1 * 2
        l1 = np.zeros(q1, dtype=np.int_)
        for i in range(q1):
            l1[i] = indicies[i]
        l2 = np.zeros(q1, dtype=np.int_)
        c1 = 0
        for i in range(q1, q2):
            bins[indicies[i]] = 1
            l2[c1] = indicies[i]
            c1 += 1
        l3 = np.zeros(len(indicies) - q2, dtype=np.int_)
        c2 = 0
        for i in range(q2, len(indicies)):
            bins[indicies[i]] = 2
            l3[c2] = indicies[i]
            c2 += 1

        found_last = False
        nsum = 0
        for i in range(3):
            if i == 0:
                o = l1
            elif i == 1:
                o = l2
            else:
                o = l3

            if not found_last:
                for n in range(len(o)):
                    if o[n] == len(X) - 1:
                        o = np.delete(o, n)
                        break

            for n in range(3):
                nsum2 = 0

                for v in o:
                    if bins[v + 1] == n:
                        nsum2 += 1

                if nsum2 > 0:
                    nsum2 /= len(X) - 1
                    nsum += nsum2 * np.log(nsum2)

        return -nsum

    @staticmethod
    def FC_LocalSimple_mean1_tauresrat(X, acfz):
        # Change in correlation length after iterative differencing
        if len(X) < 2:
            return 0
        res = _local_simple_mean(X, 1)
        mean = np.mean(res)

        nfft = int(np.power(2, np.ceil(np.log(len(res)) / np.log(2))))
        fft = np.fft.fft(res - mean, n=nfft)
        ac = _autocorr(res, fft)

        return _ac_first_zero(ac) / acfz

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def CO_Embed2_Dist_tau_d_expfit_meandiff(X, acfz):
        # Exponential fit to successive distances in 2-d embedding space
        tau = acfz
        if tau > len(X) / 10:
            tau = int(len(X) / 10)

        d = np.zeros(len(X) - tau - 1)
        d_mean = 0
        for i in range(len(d)):
            n = np.sqrt(
                np.power(X[i + 1] - X[i], 2) + np.power(X[i + tau + 1] - X[i + tau], 2)
            )
            d[i] = n
            d_mean += n
        d_mean /= len(X) - tau - 1

        smin = np.min(d)
        smax = np.max(d)
        srange = smax - smin
        std = np.std(d)

        if std == 0:
            return np.nan

        num_bins = int(
            np.ceil(srange / (3.5 * np.std(d) / np.power(len(d), 0.3333333333333333)))
        )

        if num_bins == 0:
            return np.nan
        bin_width = srange / num_bins

        histogram = np.zeros(num_bins)
        for val in d:
            idx = int((val - smin) / bin_width)
            if idx >= num_bins:
                idx = num_bins - 1
            histogram[idx] += 1

        sum = 0
        for i in range(num_bins):
            center = ((smin + bin_width * i) * 2 + bin_width) / 2
            n = np.exp(-center / d_mean) / d_mean
            if n < 0:
                n = 0

            sum += np.abs(histogram[i] / len(d) - n)

        return sum / num_bins

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(X):
        # Proportion of slower timescale fluctuations that scale with DFA (50%
        # sampling)
        cs = np.zeros(int(len(X) / 2))
        cs[0] = X[0]
        for i in range(1, len(cs)):
            cs[i] = cs[i - 1] + X[i * 2]

        return _fluct_prop(cs, len(X), True)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(X):
        # Proportion of slower timescale fluctuations that scale with linearly rescaled
        # range fits
        cs = np.zeros(len(X))
        cs[0] = X[0]
        for i in range(1, len(X)):
            cs[i] = cs[i - 1] + X[i]

        return _fluct_prop(cs, len(X), False)

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def SB_TransitionMatrix_3ac_sumdiagcov(X, acfz):
        # Trace of covariance of transition matrix between symbols in 3-letter alphabet
        ds = np.zeros(int((len(X) - 1) / acfz + 1))
        for i in range(len(ds)):
            ds[i] = X[i * acfz]
        indicies = np.argsort(ds)

        bins = np.zeros(len(ds), dtype=np.int32)
        q1 = int(len(ds) / 3)
        q2 = q1 * 2
        for i in range(q1 + 1, q2 + 1):
            bins[indicies[i]] = 1
        for i in range(q2 + 1, len(indicies)):
            bins[indicies[i]] = 2

        t = np.zeros((3, 3))
        for i in range(len(ds) - 1):
            t[bins[i + 1]][bins[i]] += 1
        t /= len(ds) - 1

        means = np.zeros(3)
        for i in range(3):
            means[i] = np.mean(t[i])

        cov = np.zeros((3, 3))
        for i in range(3):
            for n in range(3):
                covariance = 0
                for j in range(3):
                    covariance += (t[i][j] - means[i]) * (t[n][j] - means[n])
                covariance /= 2

                cov[i][n] = covariance
                cov[n][i] = covariance

        ssum = 0
        for i in range(3):
            ssum += cov[i][i]

        return ssum

    @staticmethod
    #@njit(fastmath=True, cache=True)
    def PD_PeriodicityWang_th0_01(X):
        # Periodicity measure of (Wang et al. 2007)
        y_spline = _spline_fit(X)

        y_sub = np.zeros(len(X))
        for i in range(len(X)):
            y_sub[i] = X[i] - y_spline[i]

        acmax = int(np.ceil(len(X) / 3.0))
        acf = np.zeros(acmax)
        for tau in range(1, acmax + 1):
            covariance = 0
            for i in range(len(X) - tau):
                covariance += y_sub[i] * y_sub[i + tau]
            acf[tau - 1] = covariance / (len(X) - tau)

        troughs = np.zeros(acmax, dtype=np.int32)
        peaks = np.zeros(acmax, dtype=np.int32)
        n_troughs = 0
        n_peaks = 0
        for i in range(1, acmax - 1):
            slope_in = acf[i] - acf[i - 1]
            slope_out = acf[i + 1] - acf[i]

            if slope_in < 0 and slope_out > 0:
                troughs[n_troughs] = i
                n_troughs += 1
            elif slope_in > 0 and slope_out < 0:
                peaks[n_peaks] = i
                n_peaks += 1

        out = 0
        for i in range(n_peaks):
            j = -1
            while troughs[j + 1] < peaks[i] and j + 1 < n_troughs:
                j += 1

            if j == -1 or acf[peaks[i]] - acf[troughs[j]] < 0.01 or acf[peaks[i]] < 0:
                continue

            out = peaks[i]
            break

        return out


#@njit(fastmath=True, cache=True)
def _histogram_mode(X, num_bins, smin, smax):
    bin_width = (smax - smin) / num_bins

    if bin_width == 0:
        return np.nan

    histogram = np.zeros(num_bins)
    for val in X:
        idx = int((val - smin) / bin_width)
        idx = num_bins - 1 if idx >= num_bins else idx
        histogram[idx] += 1

    edges = np.zeros(num_bins + 1, dtype=np.float32)
    for i in range(len(edges)):
        edges[i] = i * bin_width + smin

    max_count = 0
    num_maxs = 1
    max_sum = 0
    for i in range(num_bins):
        v = (edges[i] + edges[i + 1]) / 2
        if histogram[i] > max_count:
            max_count = histogram[i]
            num_maxs = 1
            max_sum = v
        elif histogram[i] == max_count:
            num_maxs += 1
            max_sum += v

    return max_sum / num_maxs


#@njit(fastmath=True, cache=True)
def _long_stretch(X_binary, val):
    last_val = 0
    max_stretch = 0
    for i in range(len(X_binary)):
        if X_binary[i] != val or i == len(X_binary) - 1:
            stretch = i - last_val
            if stretch > max_stretch:
                max_stretch = stretch
            last_val = i

    return max_stretch


#@njit(fastmath=True, cache=True)
def _outlier_include(X):
    total = 0
    threshold = 0
    for v in X:
        if v >= 0:
            total += 1
            if v > threshold:
                threshold = v

    if threshold < 0.01:
        return 0

    num_thresholds = int(threshold / 0.01) + 1
    means = np.zeros(num_thresholds)
    dists = np.zeros(num_thresholds)
    medians = np.zeros(num_thresholds)
    for i in range(num_thresholds):
        d = i * 0.01

        count = 0
        r = np.zeros(len(X))
        for n in range(len(X)):
            if X[n] >= d:
                r[count] = n + 1
                count += 1

        if count == 0:
            continue

        diff = np.zeros(count - 1)
        for n in range(len(diff)):
            diff[n] = r[n + 1] - r[n]

        means[i] = np.mean(diff) if len(diff) > 0 else 9999999999
        dists[i] = len(diff) * 100 / total
        medians[i] = np.median(r[:count]) / (len(X) / 2) - 1

    mj = 0
    fbi = num_thresholds - 1
    for i in range(num_thresholds):
        if dists[i] > 2:
            mj = i
        if means[i] == 9999999999:
            fbi = num_thresholds - 1 - i

    trim_limit = max(mj, fbi)

    return np.median(medians[: trim_limit + 1])


def _autocorr(X, X_fft):
    ca = np.fft.ifft(_multiply_complex_arr(X_fft))
    return _get_acf(X, ca)


#@njit(fastmath=True, cache=True)
def _multiply_complex_arr(X_fft):
    c = np.zeros(len(X_fft), dtype=np.complex128)
    for i, n in enumerate(X_fft):
        c[i] = n * (n.real + 1j * -n.imag)
    return c


#@njit(fastmath=True, cache=True)
def _get_acf(X, ca):
    acf = np.zeros(len(X))
    if ca[0].real != 0:
        for i in range(len(X)):
            acf[i] = ca[i].real / ca[0].real
    return acf


#@njit(fastmath=True, cache=True)
def _summaries_welch_rect(X, centroid, X_fft):
    new_length = int(len(X_fft) / 2) + 1
    p = np.zeros(new_length)
    pi2 = 2 * math.pi
    p[0] = (np.power(_complex_magnitude(X_fft[0]), 2) / len(X)) / pi2
    for i in range(1, new_length - 1):
        p[i] = ((np.power(_complex_magnitude(X_fft[i]), 2) / len(X)) * 2) / pi2
    p[new_length - 1] = (
        np.power(_complex_magnitude(X_fft[new_length - 1]), 2) / len(X)
    ) / pi2

    w = np.zeros(new_length)
    a = 1.0 / len(X_fft)
    for i in range(0, new_length):
        w[i] = i * a * math.pi * 2

    if centroid:
        cs = np.zeros(new_length)
        cs[0] = p[0]
        for i in range(1, new_length):
            cs[i] = cs[i - 1] + p[i]

        threshold = cs[new_length - 1] / 2
        for i in range(1, new_length):
            if cs[i] > threshold:
                return w[i]
        return np.nan
    else:
        tau = int(np.floor(new_length / 5))
        nsum = 0
        for i in range(tau):
            nsum += p[i]

        return nsum * (w[1] - w[0])


#@njit(fastmath=True, cache=True)
def _complex_magnitude(c):
    return np.sqrt(c.real * c.real + c.imag * c.imag)


#@njit(fastmath=True, cache=True)
def _local_simple_mean(X, train_length):
    res = np.zeros(len(X) - train_length)
    for i in range(len(res)):
        nsum = 0
        for n in range(train_length):
            nsum += X[i + n]
        res[i] = X[i + train_length] - nsum / train_length
    return res


#@njit(fastmath=True, cache=True)
def _ac_first_zero(X_ac):
    for i in range(1, len(X_ac)):
        if X_ac[i] <= 0:
            return i

    return len(X_ac)


#@njit(fastmath=True, cache=True)
def _fluct_prop(X, og_length, dfa):
    a = np.zeros(50, dtype=np.int_)
    a[0] = 5
    n_tau = 1
    smin = 1.6094379124341003  # Math.log(5);
    smax = np.log(og_length / 2)
    inc = (smax - smin) / 49
    for i in range(1, 50):
        val = int(np.round(np.exp(smin + inc * i) + 0.000000000001))
        if val != a[n_tau - 1]:
            a[n_tau] = val
            n_tau += 1

    if n_tau < 12:
        return np.nan

    f = np.zeros(n_tau)
    for i in range(n_tau):
        tau = a[i]
        buff_size = int(len(X) / tau)
        lag = 0
        if buff_size == 0:
            buff_size = 1
            lag = 1

        buffer = np.zeros((buff_size, tau))
        count = 0
        for n in range(buff_size):
            for j in range(tau - lag):
                buffer[n][j] = X[count]
                count += 1

        d = np.zeros(tau)
        for n in range(tau):
            d[n] = n + 1

        for n in range(buff_size):
            c1, c2 = _linear_regression(d, buffer[n], tau, 0)

            for j in range(tau):
                buffer[n][j] = buffer[n][j] - (c1 * (j + 1) + c2)

            if dfa:
                for j in range(tau):
                    f[i] += buffer[n][j] * buffer[n][j]
            else:
                f[i] += np.power(np.max(buffer[n]) - np.min(buffer[n]), 2)

        if dfa:
            f[i] = np.sqrt(f[i] / (buff_size * tau))
        else:
            f[i] = np.sqrt(f[i] / buff_size)

    log_a = np.zeros(n_tau)
    log_f = np.zeros(n_tau)
    for i in range(n_tau):
        log_a[i] = np.log(a[i])
        log_f[i] = np.log(f[i])

    sserr = np.zeros(n_tau - 11)
    for i in range(6, n_tau - 5):
        c1_1, c1_2 = _linear_regression(log_a, log_f, i, 0)
        c2_1, c2_2 = _linear_regression(log_a, log_f, n_tau - i + 1, i - 1)

        sum1 = 0
        for n in range(i):
            sum1 += np.power(log_a[n] * c1_1 + c1_2 - log_f[n], 2)
        sserr[i - 6] += np.sqrt(sum1)

        sum2 = 0
        for n in range(n_tau - i + 1):
            sum2 += np.power(log_a[n + i - 1] * c2_1 + c2_2 - log_f[n + i - 1], 2)
        sserr[i - 6] += np.sqrt(sum2)

    return (np.argmin(sserr) + 6) / n_tau


#@njit(fastmath=True, cache=True)
def _linear_regression(X, y, n, lag):
    sumx = 0
    sumx2 = 0
    sumxy = 0
    sumy = 0
    for i in range(lag, n + lag):
        sumx += X[i]
        sumx2 += X[i] * X[i]
        sumxy += X[i] * y[i]
        sumy += y[i]

    denom = n * sumx2 - sumx * sumx
    if denom == 0:
        return 0, 0

    return (n * sumxy - sumx * sumy) / denom, (sumy * sumx2 - sumx * sumxy) / denom


#@njit(fastmath=True, cache=True)
def _spline_fit(X):
    breaks = np.array([0, len(X) / 2 - 1, len(X) - 1])
    h0 = np.array([breaks[1] - breaks[0], breaks[2] - breaks[1]])
    h_copy = np.array([h0[0], h0[1], h0[0], h0[1]])
    hl = np.array([h_copy[3], h_copy[2], h_copy[1]])
    hr = np.array([h_copy[0], h_copy[1], h_copy[2]])

    hlCS = np.zeros(3)
    hlCS[0] = hl[0]
    for i in range(1, 3):
        hlCS[i] = hlCS[i - 1] + hl[i]

    bl = np.zeros(3)
    for i in range(3):
        bl[i] = breaks[0] - hlCS[i]

    hrCS = np.zeros(3)
    hrCS[0] = hr[0]
    for i in range(1, 3):
        hrCS[i] = hrCS[i - 1] + hr[i]

    br = np.zeros(3)
    for i in range(3):
        br[i] = breaks[2] - hrCS[i]

    breaksExt = np.zeros(9)
    for i in range(3):
        breaksExt[i] = bl[2 - i]
        breaksExt[i + 3] = breaks[i]
        breaksExt[i + 6] = br[i]

    hExt = np.zeros(8)
    for i in range(8):
        hExt[i] = breaksExt[i + 1] - breaksExt[i]

    coeffs = np.zeros((32, 4))
    for i in range(0, 32, 4):
        coeffs[i][0] = 1

    ii = np.zeros((4, 8), dtype=np.int32)
    for i in range(8):
        ii[0][i] = i
        ii[1][i] = min(1 + i, 7)
        ii[2][i] = min(2 + i, 7)
        ii[3][i] = min(3 + i, 7)

    H = np.zeros(32)
    for i in range(32):
        H[i] = hExt[ii[i % 4][int(i / 4)]]

    for k in range(1, 4):
        for j in range(k):
            for u in range(32):
                coeffs[u][j] *= H[u] / (k - j)

        Q = np.zeros((4, 8))
        for u in range(32):
            for m in range(4):
                Q[u % 4][int(u / 4)] += coeffs[u][m]

        for u in range(8):
            for m in range(1, 4):
                Q[m][u] += Q[m - 1][u]

        for u in range(32):
            if u % 4 > 0:
                coeffs[u][k] = Q[u % 4 - 1][int(u / 4)]

        fmax = np.zeros(32)
        for i in range(8):
            for j in range(4):
                fmax[i * 4 + j] = Q[3][i]

        for j in range(k + 1):
            for u in range(32):
                coeffs[u][j] /= fmax[u]

        for i in range(29):
            for j in range(k + 1):
                coeffs[i][j] -= coeffs[3 + i][j]

        for i in range(0, 32, 4):
            coeffs[i][k] = 0

    scale = np.ones(32)
    for k in range(3):
        for i in range(32):
            scale[i] /= H[i]

        for i in range(32):
            coeffs[i][3 - (k + 1)] *= scale[i]

    jj = np.zeros((4, 2), dtype=np.int32)
    for i in range(4):
        for j in range(2):
            if i == 0:
                jj[i][j] = 4 * (1 + j)
            else:
                jj[i][j] = 3

    for i in range(1, 4):
        for j in range(2):
            jj[i][j] += jj[i - 1][j]

    coeffs_out = np.zeros((8, 4))
    for i in range(8):
        coeffs_out[i] = coeffs[jj[i % 4][int(i / 4)] - 1]

    xsB = np.zeros(len(X) * 4)
    indexB = np.zeros(len(xsB), dtype=np.int32)
    breakInd = 1
    for i in range(len(X)):
        if i >= breaks[1] and breakInd < 2:
            breakInd += 1

        for j in range(4):
            xsB[i * 4 + j] = i - breaks[breakInd - 1]
            indexB[i * 4 + j] = j + (breakInd - 1) * 4

    vB = np.zeros(len(xsB))
    for i in range(len(xsB)):
        vB[i] = coeffs_out[indexB[i]][0]

    for i in range(1, 4):
        for j in range(len(xsB)):
            vB[j] = vB[j] * xsB[j] + coeffs_out[indexB[j]][i]

    A = np.zeros(len(X) * 5)
    breakInd = 0
    for i in range(len(xsB)):
        if i / 4 >= breaks[1]:
            breakInd = 1
        A[i % 4 + breakInd + int(i / 4) * 5] = vB[i]

    AT = np.zeros(len(A))
    ATA = np.zeros(25)
    ATb = np.zeros(5)
    for i in range(len(X)):
        for j in range(5):
            AT[j * len(X) + i] = A[i * 5 + j]

    for i in range(5):
        for j in range(5):
            for k in range(len(X)):
                ATA[i * 5 + j] += AT[i * len(X) + k] * A[k * 5 + j]

    for i in range(5):
        for k in range(len(X)):
            ATb[i] += AT[i * len(X) + k] * X[k]

    AElim = np.zeros((5, 5))
    for i in range(5):
        n = i * 5
        AElim[i] = ATA[n : n + 5]

    for i in range(5):
        for j in range(i + 1, 5):
            factor = AElim[j][i] / AElim[i][i]
            ATb[j] = ATb[j] - factor * ATb[i]

            for k in range(i, 5):
                AElim[j][k] = AElim[j][k] - factor * AElim[i][k]

    x = np.zeros(5)
    for i in range(4, -1, -1):
        bMinusATemp = ATb[i]
        for j in range(i + 1, 5):
            bMinusATemp -= x[j] * AElim[i][j]

        x[i] = bMinusATemp / AElim[i][i]

    C = np.zeros((5, 8))
    for i in range(32):
        C[int(i % 4 + int(i / 4) % 2)][int(i / 4)] = coeffs_out[i % 8][int(i / 8)]

    coeffs_spline = np.zeros((2, 4))
    for j in range(8):
        coefc = int(j / 2)
        coefr = j % 2
        for i in range(5):
            coeffs_spline[coefr][coefc] += C[i][j] * x[i]

    y_out = np.zeros(len(X))
    for i in range(len(X)):
        second_half = 0 if i < breaks[1] else 1
        y_out[i] = coeffs_spline[second_half][0]

    for i in range(1, 4):
        for j in range(len(X)):
            second_half = 0 if j < breaks[1] else 1
            y_out[j] = (
                y_out[j] * (j - breaks[1] * second_half) + coeffs_spline[second_half][i]
            )

    return y_out


feature_names = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "SB_BinaryStats_diff_longstretch0",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
    "CO_trev_1_num",
    "CO_HistogramAMI_even_2_5",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_MotifThree_quantile_hh",
    "FC_LocalSimple_mean1_tauresrat",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
]

features = [
    Catch22.DN_HistogramMode_5,
    Catch22.DN_HistogramMode_10,
    Catch22.SB_BinaryStats_diff_longstretch0,
    Catch22.DN_OutlierInclude_p_001_mdrmd,
    Catch22.DN_OutlierInclude_n_001_mdrmd,
    Catch22.CO_f1ecac,
    Catch22.CO_FirstMin_ac,
    Catch22.SP_Summaries_welch_rect_area_5_1,
    Catch22.SP_Summaries_welch_rect_centroid,
    Catch22.FC_LocalSimple_mean3_stderr,
    Catch22.CO_trev_1_num,
    Catch22.CO_HistogramAMI_even_2_5,
    Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi,
    Catch22.MD_hrv_classic_pnn40,
    Catch22.SB_BinaryStats_mean_longstretch1,
    Catch22.SB_MotifThree_quantile_hh,
    Catch22.FC_LocalSimple_mean1_tauresrat,
    Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff,
    Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    Catch22.SB_TransitionMatrix_3ac_sumdiagcov,
    Catch22.PD_PeriodicityWang_th0_01,
]


if __name__ == "__main__":
    from sktime.utils._testing.panel import make_classification_problem
    import time

    known = [[-0.07564333826303482, -0.1288495659828186, 100.0517578125, 99.9651107788086, 0.18721763789653778, 99.8095932006836, 0.1097816452383995, -0.028264492750167847, 100.0560302734375, 99.83897399902344, 100.05167388916016, 0.08276569843292236, 99.90945434570312, -0.24878627061843872, 0.200486958026886, 99.84133911132812, 100.16780853271484, 99.99357604980469, 0.1975000649690628, 0.20517662167549133, 0.23373816907405853, 100.06794738769531, 99.87892150878906, 0.1891719102859497, -0.010063230991363525, 100.09374237060547, -0.031584545969963074, 100.33444213867188, 0.03033745288848877, 100.03399658203125, 100.00736236572266, 99.99215698242188, 100.0328140258789, 100.06710052490234, 0.02485033869743347, 0.07179048657417297, -0.05906634032726288, -0.05447286367416382, -0.1032184362411499, 99.91938018798828, 0.028297558426856995, 0.06727643311023712, 99.98161315917969, -0.07292437553405762, 99.95405578613281, 99.98168182373047, 100.021484375, 0.0757049173116684, 100.0519790649414, 0.029379695653915405, 0.07472434639930725, 99.85824584960938, -0.09247715771198273, 0.06415878236293793, 99.84751892089844, 99.978271484375, -0.30310001969337463, 0.04055607318878174, 99.95814514160156, 99.98004150390625, 0.13502541184425354, 0.039409711956977844, -0.07156671583652496, 0.23115034401416779, -0.02225416898727417, 99.88031005859375, 100.09521484375, 99.77802276611328, 100.07967376708984, 99.95439147949219, 100.02285766601562, 99.99515533447266, 100.08013916015625, -0.12465362250804901, 0.04384997487068176, 0.004595518112182617, 100.01469421386719, 100.04025268554688, 0.0482262521982193, 0.0013912171125411987, 0.0837000161409378, -0.11048869788646698, 100.08232116699219, -0.04729580879211426, 100.19538879394531, 100.14715576171875, -0.1285834163427353, -0.039054229855537415, 99.8763427734375, 99.79098510742188, 0.069037526845932, -0.14243702590465546, 100.236572265625, 100.12715148925781, -0.0891299694776535, 99.94514465332031, 0.11806398630142212, 100.06947326660156, 0.03125236928462982, 100.0270004272461],
            [-0.22676800191402435, 0.049020253121852875, 99.8896255493164, 100.12246704101562, 0.039114005863666534, 99.9722900390625, -0.04202410578727722, -0.20249274373054504, 99.91323852539062, 99.99275207519531, 99.89012908935547, -0.10663542151451111, 100.0670166015625, -0.07898910343647003, 0.018165327608585358, 100.04106140136719, 99.98614501953125, 100.17015075683594, 0.012831531465053558, 0.011233605444431305, 0.04356757551431656, 99.89271545410156, 100.04292297363281, 0.009888812899589539, -0.19538478553295135, 99.92533874511719, 0.12839342653751373, 100.1431884765625, -0.13506601750850677, 99.85610961914062, 100.00736236572266, 99.81790161132812, 99.84012603759766, 99.88322448730469, -0.13616596162319183, -0.09629221260547638, 0.1269439309835434, 0.15007106959819794, 0.07492353022098541, 100.10285186767578, -0.14694789052009583, -0.11608700454235077, 100.16439056396484, 0.10041283816099167, 100.12391662597656, 100.16671752929688, 99.8340072631836, -0.11558403819799423, 99.88520812988281, -0.16131730377674103, -0.12071675062179565, 100.06182861328125, 0.09395276755094528, -0.11793124675750732, 100.03099060058594, 100.16497802734375, -0.1095629557967186, -0.15231195092201233, 100.144287109375, 100.18792724609375, -0.05592658370733261, -0.15835076570510864, 0.11644486337900162, 0.04603274166584015, -0.19863860309123993, 100.0760269165039, 99.89320373535156, 99.96994018554688, 99.89305877685547, 100.14378356933594, 99.84318542480469, 100.17581176757812, 99.88040161132812, 0.06045234948396683, -0.13060078024864197, -0.17115089297294617, 99.83253479003906, 99.86566162109375, -0.13359546661376953, -0.21678803861141205, -0.09202262759208679, 0.07382998615503311, 99.88740539550781, 0.1469331979751587, 99.97605895996094, 99.95421600341797, 0.07485155761241913, 0.15125307440757751, 100.056396484375, 99.98529815673828, -0.11202690005302429, 0.05778767913579941, 100.04487609863281, 99.93248748779297, 0.12043383717536926, 100.13063049316406, -0.07920172810554504, 99.88453674316406, -0.17229364812374115, 99.84723663330078],
            [9, 9, 13, 12, 11, 9, 14, 12, 10, 9, 11, 9, 9, 10, 12, 12, 10, 10, 11, 15, 12, 14, 11, 10, 12, 11, 11, 11, 13, 10, 13, 11, 14, 14, 11, 12, 11, 17, 11, 13, 12, 11, 13, 10, 13, 11, 12, 12, 11, 11, 11, 14, 13, 12, 12, 10, 13, 11, 15, 13, 14, 13, 13, 13, 15, 14, 12, 12, 16, 10, 14, 15, 12, 11, 12, 12, 11, 13, 14, 14, 12, 15, 14, 14, 15, 13, 11, 13, 12, 15, 13, 13, 13, 15, 16, 15, 13, 12, 14, 15],
            [0.0055000000000000205, 0.008000000000000021, -0.10199999999999998, 0.02400000000000002, 0.04000000000000002, 0.05400000000000002, 0.02300000000000002, 0.007000000000000021, 0.06200000000000002, 0.03400000000000002, -0.035999999999999976, 0.01350000000000002, -0.043999999999999984, 0.004000000000000021, 0.059000000000000025, 0.01900000000000002, 0.035500000000000025, -0.01199999999999998, 0.002000000000000021, -0.03249999999999998, -0.0426666666666667, 0.025999999999999964, 0.010999999999999966, -0.04800000000000003, 0.012666666666666633, -0.0826666666666667, 0.03299999999999997, 0.04466666666666663, -0.0023333333333333665, -0.05133333333333336, 0.01500000000000002, 0.01650000000000002, -0.01724999999999998, 0.050500000000000024, 0.06700000000000002, -0.00849999999999998, 0.010500000000000021, 0.008500000000000021, -0.12449999999999999, 0.03600000000000002, 0.02500000000000005, -0.007599999999999952, 0.03960000000000005, 0.005800000000000048, 0.011800000000000048, -0.012799999999999952, 0.006400000000000048, 0.02790000000000005, 0.03600000000000005, 0.001600000000000048, -0.030000000000000034, -0.0020000000000000334, 0.014833333333333299, 0.008249999999999966, 0.006999999999999966, 0.027999999999999966, -0.0006666666666667, 0.0530833333333333, -0.0010000000000000334, 0.008666666666666633, -0.05899999999999993, 0.03342857142857151, 0.03257142857142865, 0.028000000000000077, -0.008571428571428497, -0.027857142857142785, 0.012071428571428648, 0.02971428571428579, 0.028857142857142935, -0.06599999999999993, 0.004500000000000021, -0.005812499999999979, 0.03175000000000002, 0.017500000000000022, -0.00987499999999998, -0.013374999999999979, -0.002749999999999979, -0.02849999999999998, -0.01087499999999998, -0.007499999999999979, 0.029888888888888937, 0.01511111111111116, -0.007666666666666619, -0.006666666666666619, 0.015611111111111159, 0.030777777777777827, 0.026777777777777827, -0.041333333333333284, -0.00777777777777773, -0.006888888888888841, 0.013600000000000048, -0.0018999999999999523, -0.009999999999999953, 0.013600000000000048, -0.002899999999999952, 0.005800000000000048, -0.027399999999999952, 0.01580000000000005, -0.013499999999999953, -0.014199999999999952],
            [-0.03399999999999998, -0.01199999999999998, 0.058000000000000024, 0.07000000000000002, -0.03199999999999998, 0.06400000000000003, -0.01949999999999998, 0.03900000000000002, 0.08600000000000002, -0.01399999999999998, 0.0010000000000000208, 0.02200000000000002, -0.00899999999999998, 0.010500000000000021, 0.07000000000000002, -0.035999999999999976, -0.022249999999999978, -0.01449999999999998, 0.007000000000000021, 0.042500000000000024, -0.038000000000000034, -0.0006666666666667, -0.02000000000000003, 0.0133333333333333, -0.007333333333333366, -0.0186666666666667, 0.023333333333333296, -0.025333333333333367, 0.017999999999999967, 0.004999999999999966, -0.01574999999999998, -0.007749999999999979, -0.005249999999999979, -0.04199999999999998, 0.012500000000000022, 0.010000000000000021, 0.0007500000000000208, 0.03300000000000002, 0.0037500000000000207, -0.04499999999999998, -0.07639999999999995, 0.04520000000000005, -0.0011999999999999522, -0.0007999999999999521, -0.009999999999999953, 0.03140000000000005, 0.05440000000000005, 0.002600000000000048, -0.018799999999999952, 0.014400000000000048, 0.009166666666666632, 0.029666666666666633, -0.0261666666666667, 0.045333333333333295, -0.006333333333333366, -0.017333333333333367, 0.04166666666666663, -0.026083333333333368, 0.02366666666666663, -0.028666666666666698, -0.004857142857142782, -0.005571428571428497, 0.007357142857142933, -0.02714285714285707, 0.007142857142857218, 0.03857142857142865, 0.057142857142857224, -0.012285714285714212, 0.03185714285714293, 0.08028571428571436, 0.01425000000000002, 0.01112500000000002, 0.02281250000000002, -0.0042499999999999795, -0.015499999999999979, 0.004250000000000021, -0.0029999999999999792, -0.04999999999999998, 0.0023750000000000208, 0.01125000000000002, -0.051333333333333286, 0.01011111111111116, 0.004777777777777826, 0.016444444444444494, -0.019111111111111065, -0.016333333333333287, -0.0019999999999999523, -0.0056666666666666185, 0.007333333333333382, 0.01211111111111116, -0.003399999999999952, 0.0046000000000000485, 0.009100000000000049, -0.007199999999999952, 0.01860000000000005, 0.006000000000000048, 0.03950000000000005, -0.030799999999999952, 0.02670000000000005, -0.012799999999999952],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 2, 2, 4, 1, 3, 1, 2, 2, 1, 6, 1, 1, 3, 2, 1, 4, 2, 3, 2, 3, 4, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1],
            [0.04970462921011757, 0.06261005958811894, 0.04768632272979333, 0.0451562275664693, 0.05499594854386944, 0.054246572776971105, 0.054311780527300596, 0.046340312919973696, 0.04470761116241388, 0.05072165072574599, 0.0468502897712888, 0.047786265980597, 0.047876412720376786, 0.04956188620378904, 0.051110717737245404, 0.05233242188149092, 0.0467649494993163, 0.04916169092540914, 0.046590578611353586, 0.045348928286188035, 0.048851394534935214, 0.043799667869444045, 0.05280707638300241, 0.04649069123975413, 0.04949213153140844, 0.04934321168872316, 0.04790405378946123, 0.04918061031949448, 0.046816184252746314, 0.04582267490703685, 0.05261952346853771, 0.04731155852024947, 0.04875704123875069, 0.04745474744973888, 0.04627808207416853, 0.04808116349301367, 0.051838126188245195, 0.04736642498218382, 0.050950803468485885, 0.04991572601655938, 0.052331204239512735, 0.04404331630413873, 0.04742763407321163, 0.048462251022861724, 0.05270153212985662, 0.04824055901684268, 0.049535368248151516, 0.04884763221902562, 0.047185525963293416, 0.04334369930089216, 0.04683935183534274, 0.05072780147407492, 0.04808884188760532, 0.05276707678240711, 0.0467171839928775, 0.048853749324043926, 0.04717475584360832, 0.0516602593030076, 0.04817855279872245, 0.04799271339722947, 0.04985293245645674, 0.0476213570993714, 0.05009085369265384, 0.04908317558305965, 0.05284723309125905, 0.04681348261155167, 0.04982691328225945, 0.05068292864384171, 0.0499128739002238, 0.051050818385458864, 0.04609440053925981, 0.05065742424211458, 0.04871151462501918, 0.05175476682983147, 0.049398417900059145, 0.0516044729038198, 0.05065633702598089, 0.048264415197498, 0.05160658911022807, 0.04947339533142832, 0.04678264358052806, 0.050284450158893976, 0.04810142858775945, 0.051721756348417754, 0.049402814972569625, 0.04912955619595142, 0.04984312590691885, 0.050024535513318466, 0.04761150189771428, 0.04908040181424574, 0.048959293965435156, 0.05013807011314949, 0.05269422378476771, 0.050229158674571416, 0.049761602223823524, 0.05132250792511242, 0.049701119391452823, 0.049328718873430336, 0.05067667464791213, 0.04861226600737624],
            [1.5585244804918115, 1.5462526341887264, 1.607611865704152, 1.7241944055834606, 1.6321555583103222, 1.6751070203711202, 1.4910293258248433, 1.6996507129772904, 1.552388557340269, 1.6996507129772904, 1.5984079809768381, 1.6014759425526093, 1.552388557340269, 1.5922720578252956, 1.5370487494614125, 1.533980787885641, 1.6229516735830083, 1.5370487494614125, 1.5922720578252956, 1.5953400194010667, 1.547786614976612, 1.6214176927951227, 1.556990499703926, 1.603009923340495, 1.6474953661891787, 1.5493205957644975, 1.607611865704152, 1.5677283652191252, 1.5416506918250694, 1.5723303075827821, 1.5324468070977555, 1.5677283652191252, 1.53091282630987, 1.5493205957644975, 1.6045439041283807, 1.5677283652191252, 1.5984079809768381, 1.6306215775224366, 1.55545651891604, 1.6214176927951227, 1.5800002115222105, 1.5753982691585535, 1.5838351634919245, 1.6566992509164924, 1.538582730249298, 1.6022429329465522, 1.5807672019161532, 1.556990499703926, 1.5838351634919245, 1.5976409905828952, 1.6068448753102091, 1.5723303075827821, 1.576932249946439, 1.5447186534008406, 1.547786614976612, 1.5700293364009537, 1.5976409905828952, 1.5155730184310134, 1.5846021538858672, 1.5968740001889525, 1.55545651891604, 1.6106798272799232, 1.564660403643354, 1.6007089521586666, 1.4925633066127288, 1.5838351634919245, 1.6045439041283807, 1.5807672019161532, 1.568495355613068, 1.5723303075827821, 1.576932249946439, 1.5585244804918115, 1.573864288370668, 1.5454856437947835, 1.5938060386131812, 1.556223509309983, 1.552388557340269, 1.5723303075827821, 1.6175827408254086, 1.5623594324615255, 1.586136134673753, 1.561975937264554, 1.5723303075827821, 1.5972574953859238, 1.6298545871284937, 1.589971086643467, 1.5688788508100395, 1.6018594377495807, 1.582301182704039, 1.5869031250676957, 1.5915050674313527, 1.5362817590674696, 1.5615924420675826, 1.5734807931736965, 1.6168157504314657, 1.5416506918250694, 1.581534192310096, 1.5972574953859238, 1.5811506971131246, 1.6210341975981513],
            [0.5784411341750834, 0.5795250252720429, 0.5755988727668568, 0.5916030496595809, 0.571082050474236, 0.5974551489012867, 0.5766980663013541, 0.5836369883363131, 0.5642749643581372, 0.577730332731632, 0.5775800141112541, 0.5715433005482531, 0.5843973892406207, 0.5793498200278062, 0.5673236542527417, 0.5781440764933501, 0.591226939977444, 0.5905023532366281, 0.5762582393810015, 0.5767693618043083, 0.5785579836080827, 0.5707105261381026, 0.5786193914224244, 0.5754528173396158, 0.5812201876605756, 0.5914949335701991, 0.5894920119608333, 0.5815998440124548, 0.5747736778433867, 0.5808314696735098, 0.577092655037737, 0.5859010694082286, 0.5823105871169258, 0.5657740607837198, 0.57075780922285, 0.5724888882844996, 0.583824161591746, 0.5893104491631282, 0.5743974141388343, 0.5776392839444531, 0.5744767370484309, 0.5785130068341164, 0.5817045897619544, 0.5698632766775852, 0.5725030061969549, 0.5652561946654016, 0.5814401072598266, 0.5800631758441501, 0.5753599793786842, 0.5801710348107643, 0.5727663585839601, 0.5804753502484231, 0.5655094748777394, 0.5758984876993187, 0.586028791249332, 0.5837361807079975, 0.5712198913067276, 0.5719115530257245, 0.5761462968624343, 0.5838808307179871, 0.5754687735086177, 0.5810554572212324, 0.5694268774456449, 0.5775481839571946, 0.5666296798600714, 0.5734330356659063, 0.5780699176324101, 0.5794332782608662, 0.5674489275590717, 0.578996535369198, 0.577812402039059, 0.5745320589811415, 0.5863845521293066, 0.5797883193403452, 0.5728727685980236, 0.5770503543590726, 0.5839314814102803, 0.5780279157269719, 0.5796989496966232, 0.5807228886108311, 0.5760259065884032, 0.5837404668327634, 0.57521106170937, 0.5759869145029727, 0.572020508864145, 0.5719163990305162, 0.5747378439191891, 0.5835383349584353, 0.5731966675109708, 0.5814688505781263, 0.5731989558784912, 0.5781069111033912, 0.5772376717340926, 0.5743524124227347, 0.5742634706052171, 0.5825391457411873, 0.581603162245039, 0.5760870398667488, 0.5751128707063425, 0.5756556434783586],
            [-0.015256656077128457, 0.0043893074652025825, 0.000670120110890518, 0.01520436049487334, 0.008494600799011673, -0.034664395222721606, 0.026155106936881122, 0.01835007813992413, 0.004439114318119222, 0.010705409184971955, -0.013975727065598915, -0.035001588629048924, -0.049904551567592804, -0.01824695482407708, -0.023268217844915777, 0.0030574449513848206, -0.0026313336235612712, 0.006607006008593841, -0.0025210669157245294, -0.009043974090631252, 0.004385590150228936, 0.0035579011043422727, 0.01180292439538302, -0.00026821197362010833, 0.009133624668876577, 0.007084610959763727, 0.024602178776823618, 0.004187334011487607, -0.019583464177527383, -0.010060139569291533, -0.0017534771575934483, 0.011838165440341284, 0.013630768842048049, -0.0002391313998088209, 0.0019380055136218458, -0.0046622558599519436, 0.014854874195205288, 0.0067798225616388956, -0.011170436384327017, 0.02068234244715065, -0.0020615706637181403, -0.00877770302464561, 0.005866534212472975, -0.006804311554366627, 0.00036579028590608565, 0.00516259375191506, 0.005706970585738785, 0.010773318073170666, -0.019966463792957505, -0.0063224449779900275, -0.001734733020458722, -0.003412107283850905, -0.006976740506680455, -0.0028870866687317116, 0.005692743586782012, -0.008216485461201193, -0.0008072438925153326, 0.00191674274277684, -0.00627398177944199, 0.006830396206662331, 0.0017155925984336739, 0.019051426170726794, -0.003809433806821927, 0.0024198255331115306, -0.0006224109204105434, 0.009956604813442887, 0.0023099018662737247, -0.00038096302924890755, 0.0020473387327539123, 0.012201655181553453, 0.0065772519686249946, -0.003930531638418481, -0.005781376010138614, -0.006634064221727057, 0.007113360627422839, -0.0012507482502353148, 0.012004424802785197, 0.0055994081907562165, 0.0019985190430983875, 0.0022429415088317836, -0.007842303327161413, -0.009233363277643562, 0.0014252625440793298, -0.009257863969345982, -0.001669694160166306, -0.006057156738131746, -0.0009225490091064475, 0.0026704731334588477, -0.003419907226406121, 0.0021682054603320284, 0.0031439282490764477, 0.00046587823490700317, 0.00219484949621662, -0.010281131620620571, -0.002861613458216963, -0.008124149698198897, 0.01170938634766787, 0.01216998962247645, -0.0017897318203637718, 0.0059151443648267485],
            [0.011088644737001182, 0.006106479116234229, 0.008540132190657307, 0.01215455463167552, 0.0062412429173624925, 0.0072606046759691786, 0.00391711204807907, 0.009872861525821001, 0.003932978796373394, 0.006775579625632816, 0.002999278995813938, 0.0073115952007183965, 0.0038655521296671403, 0.005763322600042477, 0.002613632292900292, 0.004215734325665405, 0.0025947499495049533, 0.004138255060933963, 0.0041960771173677365, 0.007314006280816977, 0.001686420448837351, 0.0028043124142427857, 0.002156553932013344, 0.004340031962368264, 0.0035988045999559353, 0.003443138073324137, 0.0032367233510094996, 0.0021897332114714165, 0.0024776952696137352, 0.0023692674931189593, 0.002201656245517418, 0.0031104148654291816, 0.0015627436101457574, 0.0010846509873418879, 0.00247940842157903, 0.002043964441435747, 0.0030317057578640072, 0.0010426853934614425, 0.0031605540931524618, 0.0028165416647423507, 0.0007324786819262313, 0.0016097435748517552, 0.001409505552633946, 0.0013939264599249295, 0.0011647773473151231, 0.0015606028588674473, 0.0021142942662142603, 0.0014584051867249698, 0.0034763336786002067, 0.0033188254463531355, 0.0014014891238674357, 0.0006613036345374907, 0.0015225754623512746, 0.001984260748522992, 0.002238954865521263, 0.0013323764398865206, 0.0009110823570537501, 0.001633278670788285, 0.0017721293877693601, 0.0009986033180143647, 0.0020334630219392405, 0.0013073268410399475, 0.0016378232657061733, 0.0011210814885458456, 0.0011108486093220963, 0.0011898238638687563, 0.00131748936420535, 0.0014170684592477519, 0.0011036209895212144, 0.0009395259659799148, 0.0010040639587407397, 0.0010368099946602787, 0.0010312811685473228, 0.0009577778572644951, 0.0008929300017888745, 0.0010235380229073133, 0.001042433043741681, 0.0008473201305698451, 0.0008261591434472113, 0.00043750011734836053, 0.0010288593455633488, 0.0008084660143409603, 0.0008462278997766157, 0.0011799660509322575, 0.0010932366567732493, 0.001321961812565027, 0.0006060293414950991, 0.001009794305813455, 0.0008462591890188933, 0.0007015093228450222, 0.00021144299238856276, 0.00046139733886594876, 0.000601517303119193, 0.0007304840642636489, 0.0005219675359395624, 0.0002963260627585915, 0.0004982814542779311, 0.0008803132439040077, 0.0009170317602069844, 0.0006435076249537437],
            [2, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 1, 3, 2, 2, 1, 4, 2, 2, 2, 1, 2, 1, 1, 4, 3, 3, 6, 3, 2, 3, 3, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 3, 2, 2, 1, 3, 1, 1, 2, 1, 6, 4, 3, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 5, 2, 2, 1, 5, 4, 1, 2, 2, 5, 2, 2, 2, 2, 3, 1, 1, 1, 2, 2, 3, 2, 1, 2, 1, 1, 4, 1],
            [0.9619619619619619, 0.954954954954955, 0.9579579579579579, 0.950950950950951, 0.953953953953954, 0.9579579579579579, 0.9529529529529529, 0.95995995995996, 0.9619619619619619, 0.95995995995996, 0.9534767383691846, 0.959479739869935, 0.9469734867433717, 0.9574787393696849, 0.9599799899949975, 0.9479739869934968, 0.9549774887443722, 0.9514757378689345, 0.9569784892446224, 0.9579789894947474, 0.9546515505168389, 0.9593197732577526, 0.952650883627876, 0.9489829943314438, 0.9546515505168389, 0.9603201067022341, 0.9516505501833945, 0.9543181060353451, 0.9569856618872957, 0.9599866622207403, 0.9502375593898474, 0.948737184296074, 0.9547386846711677, 0.9579894973743436, 0.9522380595148787, 0.9517379344836209, 0.9532383095773943, 0.962240560140035, 0.9504876219054764, 0.9584896224056014, 0.9581916383276655, 0.9477895579115824, 0.9537907581516303, 0.958991798359672, 0.9567913582716543, 0.9543908781756352, 0.9515903180636127, 0.9495899179835967, 0.9551910382076415, 0.956991398279656, 0.9531588598099683, 0.953492248708118, 0.9539923320553425, 0.9549924987497916, 0.9554925820970162, 0.9523253875645941, 0.9553258876479414, 0.9526587764627438, 0.9514919153192198, 0.9544924154025671, 0.9527075296470925, 0.9554222031718816, 0.9572796113730533, 0.9534219174167738, 0.952993284754965, 0.9515645092156022, 0.952993284754965, 0.9578511215887984, 0.9601371624517788, 0.954564937848264, 0.9562445305663207, 0.9567445930741343, 0.9574946868358545, 0.9548693586698337, 0.9599949993749218, 0.9571196399549944, 0.960745093136642, 0.9547443430428804, 0.9566195774471808, 0.9584948118514814, 0.9539948883209245, 0.956772974774975, 0.9548838759862207, 0.9573285920657851, 0.9561062340260029, 0.9524391599066563, 0.9556617401933548, 0.9581064562729192, 0.9526614068229803, 0.9553283698188688, 0.9554955495549555, 0.9503950395039504, 0.9512951295129513, 0.957995799579958, 0.9541954195419542, 0.9531953195319532, 0.9518951895189519, 0.9564956495649565, 0.9568956895689569, 0.9528952895289529],
            [5, 6, 7, 5, 6, 5, 5, 6, 5, 5, 6, 6, 6, 7, 5, 7, 6, 6, 6, 7, 6, 6, 7, 7, 6, 8, 6, 6, 8, 6, 8, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 8, 6, 6, 7, 6, 8, 7, 7, 6, 6, 7, 7, 7, 6, 8, 10, 7, 7, 7, 7, 6, 6, 7, 6, 7, 7, 7, 7, 6, 6, 8, 7, 7, 7, 9, 7, 6, 7, 7, 7, 7, 7, 6, 7, 7, 7, 8, 6, 9, 7, 8, 6, 7, 6, 9, 8, 7, 7],
            [2.1962061011175087, 2.194615875167326, 2.1966286804060053, 2.194242501797669, 2.1959119852379407, 2.194715598269832, 2.196426260083125, 2.1958430530017807, 2.196083094549542, 2.192041035464842, 2.197129235068252, 2.196056682691427, 2.194076882313547, 2.1968526831742676, 2.195777179116218, 2.1950419758885182, 2.1954401250213142, 2.19656936281488, 2.195267519909418, 2.195224242013793, 2.1967901824714633, 2.196949950603822, 2.19558230274153, 2.195758928204687, 2.1967649786868897, 2.1960991345370484, 2.196373293743389, 2.196182716246947, 2.1971536307674975, 2.19622922962561, 2.1966863220741284, 2.1966604156634717, 2.1965448342565748, 2.1969945079796216, 2.197090389239114, 2.196934714469845, 2.1963276355751757, 2.1960735521859878, 2.1969403318393113, 2.1967160340892864, 2.1969288333464667, 2.1966884081054356, 2.1969217277220547, 2.196014967428756, 2.1966711938406442, 2.1967560101148402, 2.1967661309493938, 2.1969472275664788, 2.1960390233595435, 2.1964003513767083, 2.1967133342217364, 2.19693772291586, 2.196984790983799, 2.1971348562947965, 2.1967111620041875, 2.1965475119428644, 2.196896842562747, 2.197014113082182, 2.1971060039404917, 2.196704472311037, 2.196382045359333, 2.196956408620042, 2.196997914565055, 2.1969425941671386, 2.1959907096389664, 2.1971695327199643, 2.196992902344908, 2.197074121285789, 2.1970981562511698, 2.196991657582012, 2.1971114530598257, 2.1968700439082536, 2.1971692379916212, 2.196906809505881, 2.1968513815359128, 2.1971705283534795, 2.1968570104936793, 2.197107872229829, 2.196989613123543, 2.1971099083936894, 2.197068230704794, 2.1969502857944176, 2.197044636825541, 2.196998303742788, 2.197180521917175, 2.1970130590045547, 2.1972131994718063, 2.1969482857387335, 2.196940301084963, 2.1970517051962544, 2.197045876950111, 2.1970870511903584, 2.196890825651289, 2.197127237054817, 2.1971650794202966, 2.1970351508292514, 2.196671537952103, 2.19679762382266, 2.197018992553525, 2.197064346054422],
            [1.0, 0.25, 1.0, 1.0, 1.0, 1.0, 0.14285714285714285, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.3333333333333333, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.3333333333333333, 1.0, 0.2, 1.0, 1.0, 1.0, 0.16666666666666666, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.16666666666666666, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 0.3333333333333333, 1.0, 0.25, 1.0, 0.14285714285714285, 1.0, 1.0, 1.0, 0.5, 0.3333333333333333, 1.0, 0.25, 1.0, 0.2, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.3333333333333333, 1.0, 1.0, 1.0, 1.0, 0.5, 0.16666666666666666, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            [0.2300453859258666, 0.25247063177954854, 0.24596909133925027, 0.23855772347181842, 0.30203632669893155, 0.2548113934273738, 0.3051060687663565, 0.22992291563007897, 0.3000459586889097, 0.28277299326269667, 0.2187959926143159, 0.24357021068344836, 0.2578335284074818, 0.24596577089255156, 0.2867933430082778, 0.2469895372891057, 0.24648243364539835, 0.2784425728070021, 0.25574138530874085, 0.27256171677704255, 0.30667614811651694, 0.26269372828457505, 0.30468607065348785, 0.2949171995374715, 0.24581114365286785, 0.250123038975584, 0.2651177665819501, 0.20833629841122203, 0.28563177110580396, 0.2967845451883759, 0.29695965785246403, 0.2975530851638281, 0.30097842243458733, 0.31122099640692397, 0.26968222791951824, 0.273254425782179, 0.23365144344781344, 0.25465729047673613, 0.30313511496197715, 0.2415856939286508, 0.30663838744385097, 0.2551244696339735, 0.25573432360454024, 0.2650405903914317, 0.3051290658752833, 0.27339213442874016, 0.26562699751766783, 0.2852067587889056, 0.2728311617015359, 0.23697471880823823, 0.26986632049309717, 0.27391996811005165, 0.30405111437287285, 0.29894558697176105, 0.30184021527664007, 0.2913151225696284, 0.24212631792955236, 0.2879375423771719, 0.2656133972040211, 0.23935580804168521, 0.2517640128739765, 0.2549264687818927, 0.3023501560251229, 0.2600113338818126, 0.31617411208663615, 0.2576418012187987, 0.2446005181992934, 0.2534989843848779, 0.2878681352671877, 0.30050343329251517, 0.242883441684842, 0.29186666428602415, 0.26369263771554646, 0.2962476043685415, 0.24003512150960418, 0.31661635532226523, 0.29291092216813497, 0.24262534394682467, 0.283562855605459, 0.29663428652429547, 0.27345613993764184, 0.27389465841057037, 0.2681420594067151, 0.264706103998558, 0.21511954139652215, 0.2608539503039714, 0.284683943449041, 0.26143190075211914, 0.26554216187385027, 0.26289444128545375, 0.26381459069248175, 0.28579234827332795, 0.30030099253475373, 0.2747255658753526, 0.23927760666373588, 0.29856528684788947, 0.2584829816489029, 0.2790733175908464, 0.22778569853075872, 0.2607997008340703],
            [0.8125, 0.8125, 0.8333333333333334, 0.14583333333333334, 0.4791666666666667, 0.4791666666666667, 0.875, 0.8541666666666666, 0.125, 0.5833333333333334, 0.7142857142857143, 0.7959183673469388, 0.8163265306122449, 0.12244897959183673, 0.7959183673469388, 0.6326530612244898, 0.12244897959183673, 0.7755102040816326, 0.7755102040816326, 0.2857142857142857, 0.22448979591836735, 0.5918367346938775, 0.5918367346938775, 0.14285714285714285, 0.12244897959183673, 0.8367346938775511, 0.12244897959183673, 0.7959183673469388, 0.7551020408163265, 0.8775510204081632, 0.8163265306122449, 0.7551020408163265, 0.12244897959183673, 0.7551020408163265, 0.5510204081632653, 0.16326530612244897, 0.14285714285714285, 0.7959183673469388, 0.4897959183673469, 0.8775510204081632, 0.8775510204081632, 0.20408163265306123, 0.12244897959183673, 0.14285714285714285, 0.5510204081632653, 0.12244897959183673, 0.7755102040816326, 0.5714285714285714, 0.14285714285714285, 0.5918367346938775, 0.673469387755102, 0.12244897959183673, 0.6530612244897959, 0.6122448979591837, 0.22448979591836735, 0.8775510204081632, 0.7346938775510204, 0.5102040816326531, 0.14285714285714285, 0.46938775510204084, 0.12244897959183673, 0.8775510204081632, 0.8367346938775511, 0.5102040816326531, 0.8775510204081632, 0.12244897959183673, 0.673469387755102, 0.14285714285714285, 0.5918367346938775, 0.8775510204081632, 0.12, 0.54, 0.2, 0.86, 0.84, 0.7, 0.14, 0.88, 0.22, 0.84, 0.12, 0.88, 0.26, 0.12, 0.48, 0.12, 0.12, 0.16, 0.12, 0.6, 0.52, 0.12, 0.7, 0.12, 0.12, 0.72, 0.78, 0.12, 0.88, 0.54],
            [0.125, 0.3333333333333333, 0.125, 0.20833333333333334, 0.2916666666666667, 0.14583333333333334, 0.22916666666666666, 0.14583333333333334, 0.1875, 0.3333333333333333, 0.8571428571428571, 0.2653061224489796, 0.14285714285714285, 0.2857142857142857, 0.16326530612244897, 0.1836734693877551, 0.5306122448979592, 0.22448979591836735, 0.16326530612244897, 0.3673469387755102, 0.2857142857142857, 0.12244897959183673, 0.16326530612244897, 0.2857142857142857, 0.12244897959183673, 0.6122448979591837, 0.16326530612244897, 0.12244897959183673, 0.12244897959183673, 0.12244897959183673, 0.20408163265306123, 0.1836734693877551, 0.12244897959183673, 0.20408163265306123, 0.3469387755102041, 0.42857142857142855, 0.22448979591836735, 0.12244897959183673, 0.2857142857142857, 0.24489795918367346, 0.24489795918367346, 0.40816326530612246, 0.12244897959183673, 0.1836734693877551, 0.2653061224489796, 0.12244897959183673, 0.30612244897959184, 0.8163265306122449, 0.20408163265306123, 0.12244897959183673, 0.1836734693877551, 0.3673469387755102, 0.6938775510204082, 0.3877551020408163, 0.42857142857142855, 0.14285714285714285, 0.16326530612244897, 0.1836734693877551, 0.20408163265306123, 0.2857142857142857, 0.40816326530612246, 0.1836734693877551, 0.20408163265306123, 0.12244897959183673, 0.16326530612244897, 0.12244897959183673, 0.22448979591836735, 0.20408163265306123, 0.5714285714285714, 0.16326530612244897, 0.38, 0.74, 0.66, 0.2, 0.12, 0.2, 0.3, 0.2, 0.16, 0.2, 0.12, 0.2, 0.12, 0.22, 0.2, 0.22, 0.14, 0.18, 0.3, 0.18, 0.12, 0.18, 0.78, 0.5, 0.22, 0.28, 0.58, 0.26, 0.12, 0.12],
            [0.00011422834245657066, 0.0011558953350214783, 4.776214319090528e-05, 0.00031028693024021705, 0.0001619904856474761, 0.000316298948264247, 0.0012563644779474972, 0.00015698047062745097, 0.00014595843758339583, 0.0005731457182908632, 7.590922357960717e-06, 0.00012871201316849853, 0.0003390890042319806, 4.262594862547162e-05, 9.08482723631205e-05, 0.00025851677503329154, 0.00021096090816794075, 0.00011289233845123059, 0.00022647641980069587, 0.0002277276707388212, 0.00012253369416984373, 2.9464083893252802e-05, 0.00012893619322128352, 0.0001628122270981884, 5.477725282292781e-05, 0.00011774515035077277, 9.213548749512776e-05, 0.00011552144590599889, 6.596989852828949e-06, 9.999257653332848e-05, 0.00022347500499947, 5.7195260722324085e-05, 0.00026126119588058177, 0.00019077405969951784, 1.5049190321252926e-05, 3.268300612704237e-05, 0.00010415623827603979, 0.00012406202325775245, 0.0001380690258836278, 5.581957296442564e-05, 5.511016522027522e-05, 6.086434330274743e-05, 3.2372947884235764e-05, 0.0001428571371405708, 0.0002646917110181408, 5.162064619365167e-05, 5.3861542462523324e-05, 8.11849349583771e-05, 0.0001325997012431759, 9.01027041442162e-05, 5.818606040385523e-05, 3.0287872894079307e-05, 9.917721823691146e-05, 2.605207810413012e-05, 2.7648058596687547e-05, 0.00012878954909314918, 3.461338868659771e-05, 3.1724850078772826e-05, 1.213367385053333e-05, 5.671334731817972e-05, 8.439145706137558e-05, 3.0879570403525486e-05, 0.00011724960752437369, 3.1185780334807595e-05, 0.00023647270894518145, 6.362361905530261e-06, 2.700091127395081e-05, 1.654894561951825e-05, 8.440196430052846e-06, 9.272036719225156e-05, 1.1872759671072568e-05, 1.5598928361963227e-05, 6.074435180548713e-06, 0.0001363683238025433, 4.04111438211979e-05, 8.879439164617308e-06, 3.0515255720656932e-05, 1.3060556601745892e-05, 2.501667044339208e-05, 3.533016287330153e-05, 1.6287569876175317e-05, 3.0416809260898943e-05, 1.924295910313859e-05, 2.510434386524978e-05, 4.552451928999216e-06, 2.4869723829125945e-05, 0.0002393817834798995, 3.0920451076529944e-05, 3.235698428016534e-05, 1.760473499125646e-05, 2.1124224633684527e-05, 6.266506351880497e-05, 0.00016837467279829122, 1.0142028304240562e-05, 6.748016202426976e-06, 4.936641125651283e-05, 6.1018869830444086e-05, 4.634593539028536e-05, 2.076081862278303e-05, 1.7750216532470974e-05],
            [7, 5, 4, 3, 5, 14, 3, 6, 7, 6, 18, 9, 4, 2, 9, 6, 3, 3, 13, 5, 9, 3, 17, 4, 5, 8, 3, 6, 4, 12, 8, 12, 6, 10, 6, 13, 8, 32, 6, 12, 8, 16, 5, 31, 7, 60, 22, 6, 35, 5, 34, 28, 10, 13, 3, 3, 55, 56, 48, 8, 40, 11, 36, 34, 274, 40, 11, 71, 85, 23, 139, 76, 14, 84, 65, 46, 102, 133, 17, 86, 59, 136, 24, 684, 84, 91, 48, 51, 231, 13, 355, 119, 107, 84, 22, 177, 117, 30, 64, 61]]

    cases = []
    for i in range(1000, 10001, 1000):
        X, _ = make_classification_problem(
            n_instances=10,
            n_columns=1,
            n_timepoints=i,
            n_classes=2,
            return_numpy=True,
            random_state=i,
        )
        X = np.reshape(X, (X.shape[0], -1))
        for n in X:
            cases.append(n)

    c22 = Catch22(outlier_norm=True)
    for i, feature in enumerate(feature_names):
        time_sum = 0
        for n, series in enumerate(cases):
            start = time.time_ns()
            v = c22._transform_case_single(series, i)
            end = time.time_ns()
            np.testing.assert_array_almost_equal(v, known[i][n])
            time_sum += end - start
        print(f"{feature}: {time_sum*1e-9} seconds")
