# -*- coding: utf-8 -*-
"""Catch22 features.

A transformer for the Catch22 features.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["Catch22"]

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sktime.datatypes import convert_to
from sktime.transformations.base import BaseTransformer
from sktime.utils.validation import check_n_jobs

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


def _verify_features(features, catch24):
    if isinstance(features, str):
        if features == "all":
            f_idx = [i for i in range(22)]
            if catch24:
                f_idx += [22, 23]
        elif features in feature_names:
            f_idx = [feature_names.index(features)]
        elif catch24 and features == "Mean":
            f_idx = [22]
        elif catch24 and features == "StandardDeviation":
            f_idx = [23]
        else:
            raise ValueError("Invalid feature selection.")
    elif isinstance(features, int):
        if features >= 0 and features < 22:
            f_idx = [features]
        elif catch24 and features == 22:
            f_idx = [22]
        elif catch24 and features == 23:
            f_idx = [23]
        else:
            raise ValueError("Invalid feature selection.")
    elif isinstance(features, (list, tuple)):
        if len(features) > 0:
            f_idx = []
            for f in features:
                if isinstance(f, str):
                    if f in feature_names:
                        f_idx.append(feature_names.index(f))
                    elif catch24 and f == "Mean":
                        f_idx.append(22)
                    elif catch24 and f == "StandardDeviation":
                        f_idx.append(23)
                    else:
                        raise ValueError("Invalid feature selection.")
                elif isinstance(f, int):
                    if f >= 0 and f < 22:
                        f_idx.append(f)
                    elif catch24 and f == 22:
                        f_idx.append(22)
                    elif catch24 and f == 23:
                        f_idx.append(23)
                    else:
                        raise ValueError("Invalid feature selection.")
                else:
                    raise ValueError("Invalid feature selection.")
        else:
            raise ValueError("Invalid feature selection.")
    else:
        raise ValueError("Invalid feature selection.")

    return f_idx


class Catch22(BaseTransformer):
    """Canonical Time-series Characteristics (Catch22).

    Overview: Input n series with d dimensions of length m
    Transforms series into the 22 Catch22 [1]_ features extracted from the hctsa [2]_
    toolbox.

    Parameters
    ----------
    features : int/str or List of int/str, optional, default="all"
        The Catch22 features to extract by feature index, feature name as a str or as a
        list of names or indices for multiple features. If "all", all features are
        extracted.
        Valid features are as follows:
            ["DN_HistogramMode_5", "DN_HistogramMode_10",
            "SB_BinaryStats_diff_longstretch0", "DN_OutlierInclude_p_001_mdrmd",
            "DN_OutlierInclude_n_001_mdrmd", "CO_f1ecac", "CO_FirstMin_ac",
            "SP_Summaries_welch_rect_area_5_1", "SP_Summaries_welch_rect_centroid",
            "FC_LocalSimple_mean3_stderr", "CO_trev_1_num", "CO_HistogramAMI_even_2_5",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi", "MD_hrv_classic_pnn40",
            "SB_BinaryStats_mean_longstretch1", "SB_MotifThree_quantile_hh",
            "FC_LocalSimple_mean1_tauresrat", "CO_Embed2_Dist_tau_d_expfit_meandiff",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SB_TransitionMatrix_3ac_sumdiagcov", "PD_PeriodicityWang_th0_01"]
    catch24 : bool, optional, default=False
        Extract the mean and standard deviation as well as the 22 Catch22 features if
        true. If a List of specific features to extract is provided, "Mean" and/or
        "StandardDeviation" must be added to the List to extract these features.
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features, which can take a
        while to process for large values.
    replace_nans : bool, optional, default=True
        Replace NaN or inf values from the Catch22 transform with 0.
    n_jobs : int, optional, default=1
        The number of jobs to run in parallel for transform. Requires multiple input
        cases. A value of -1 uses all CPU cores.

    See Also
    --------
    Catch22Classifier

    Notes
    -----
    Original Catch22 package implementations:
    https://github.com/DynamicsAndNeuralSystems/Catch22

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/transformers/Catch22.java

    References
    ----------
    .. [1] Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., &
    Jones, N. S. (2019). catch22: Canonical time-series characteristics. Data Mining
    and Knowledge Discovery, 33(6), 1821-1852.
    .. [2] Fulcher, B. D., Little, M. A., & Jones, N. S. (2013). Highly comparative
    time-series analysis: the empirical structure of time series and their methods.
    Journal of the Royal Society Interface, 10(83), 20130048.
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "nested_univ",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "python_dependencies": "numba",
    }

    def __init__(
        self,
        features="all",
        catch24=False,
        outlier_norm=False,
        replace_nans=False,
        n_jobs=1,
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.n_jobs = n_jobs

        self.features_arguments = (
            features
            if features != "all"
            else (
                feature_names + ["Mean", "StandardDeviation"]
                if catch24
                else feature_names
            )
        )

        if isinstance(features, str):
            if features == "all":
                self.n_transformed_features = 24 if catch24 else 22
            else:
                self.n_transformed_features = 1
        elif isinstance(features, (list, tuple)):
            self.n_transformed_features = len(features)
        else:
            raise ValueError("features must be a str, list or tuple")

        self._transform_features = None

        # todo remove in v0.16
        self._case_id = None
        self._st_n_instances = 0
        self._st_series_length = 0
        self._outlier_series = None
        self._smin = None
        self._smax = None
        self._smean = None
        self._fft = None
        self._ac = None
        self._acfz = None

        super(Catch22, self).__init__()

    def _transform(self, X, y=None):
        """Transform data into the Catch22 features.

        Parameters
        ----------
        X : 3D numpy array of shape [n_instances, n_dimensions, n_features],
            input time series panel.
        y : ignored.

        Returns
        -------
        c22 : Pandas DataFrame of shape [n_instances, c*n_dimensions] where c is the
             number of features requested, containing Catch22 features for X.
        """
        n_instances = X.shape[0]

        f_idx = _verify_features(self.features, self.catch24)

        threads_to_use = check_n_jobs(self.n_jobs)

        # todo remove in v0.16 and add to docstring: ``-1`` means using all processors.
        if self.n_jobs == -1:
            threads_to_use = 1
            warnings.warn(
                "``n_jobs`` default was changed to 1 from -1 in version 0.14.0. "
                "In version 0.16.0 a value of -1 will use all CPU cores instead of the "
                "current 1 CPU core."
            )

        c22_list = Parallel(n_jobs=threads_to_use)(
            delayed(self._transform_case)(
                X.iloc[i],
                f_idx,
            )
            for i in range(n_instances)
        )

        if self.replace_nans:
            c22_list = np.nan_to_num(c22_list, False, 0, 0, 0)

        return pd.DataFrame(c22_list)

    def _get_feature_fun(self, idx):
        """Get the idx-th feature generation function from catch22_numba."""
        from sktime.transformations.panel._catch22_numba import (
            _CO_Embed2_Dist_tau_d_expfit_meandiff,
            _CO_f1ecac,
            _CO_FirstMin_ac,
            _CO_HistogramAMI_even_2_5,
            _CO_trev_1_num,
            _DN_HistogramMode_5,
            _DN_HistogramMode_10,
            _DN_OutlierInclude_n_001_mdrmd,
            _DN_OutlierInclude_p_001_mdrmd,
            _FC_LocalSimple_mean1_tauresrat,
            _FC_LocalSimple_mean3_stderr,
            _IN_AutoMutualInfoStats_40_gaussian_fmmi,
            _MD_hrv_classic_pnn40,
            _PD_PeriodicityWang_th0_01,
            _SB_BinaryStats_diff_longstretch0,
            _SB_BinaryStats_mean_longstretch1,
            _SB_MotifThree_quantile_hh,
            _SB_TransitionMatrix_3ac_sumdiagcov,
            _SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
            _SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
            _SP_Summaries_welch_rect_area_5_1,
            _SP_Summaries_welch_rect_centroid,
        )

        features = [
            _DN_HistogramMode_5,
            _DN_HistogramMode_10,
            _SB_BinaryStats_diff_longstretch0,
            _DN_OutlierInclude_p_001_mdrmd,
            _DN_OutlierInclude_n_001_mdrmd,
            _CO_f1ecac,
            _CO_FirstMin_ac,
            _SP_Summaries_welch_rect_area_5_1,
            _SP_Summaries_welch_rect_centroid,
            _FC_LocalSimple_mean3_stderr,
            _CO_trev_1_num,
            _CO_HistogramAMI_even_2_5,
            _IN_AutoMutualInfoStats_40_gaussian_fmmi,
            _MD_hrv_classic_pnn40,
            _SB_BinaryStats_mean_longstretch1,
            _SB_MotifThree_quantile_hh,
            _FC_LocalSimple_mean1_tauresrat,
            _CO_Embed2_Dist_tau_d_expfit_meandiff,
            _SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
            _SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
            _SB_TransitionMatrix_3ac_sumdiagcov,
            _PD_PeriodicityWang_th0_01,
        ]

        return features[idx]

    def _transform_case(self, X, f_idx):
        from sktime.transformations.panel._catch22_numba import (
            _ac_first_zero,
            _autocorr,
            _normalise_series,
        )

        c22 = np.zeros(len(f_idx) * len(X))

        if self._transform_features is not None and len(
            self._transform_features
        ) == len(c22):
            transform_feature = self._transform_features
        else:
            transform_feature = [True] * len(c22)

        f_count = -1
        for i in range(len(X)):
            series = np.array(X[i])
            dim = i * len(f_idx)
            outlier_series = None
            smin = None
            smax = None
            smean = None
            fft = None
            ac = None
            acfz = None

            for n, feature in enumerate(f_idx):
                f_count += 1
                if not transform_feature[f_count]:
                    continue

                args = [series]

                if feature == 0 or feature == 1 or feature == 11:
                    if smin is None:
                        smin = np.min(series)
                    if smax is None:
                        smax = np.max(series)
                    args = [series, smin, smax]
                elif feature == 2 or feature == 22:
                    if smean is None:
                        smean = np.mean(series)
                    args = [series, smean]
                elif feature == 3 or feature == 4:
                    if self.outlier_norm:
                        if smean is None:
                            smean = np.mean(series)
                        if outlier_series is None:
                            outlier_series = _normalise_series(series, smean)
                        args = [outlier_series]
                    else:
                        args = [series]
                elif feature == 7 or feature == 8:
                    if smean is None:
                        smean = np.mean(series)
                    if fft is None:
                        nfft = int(
                            np.power(2, np.ceil(np.log(len(series)) / np.log(2)))
                        )
                        fft = np.fft.fft(series - smean, n=nfft)
                    args = [series, fft]
                elif feature == 5 or feature == 6 or feature == 12:
                    if smean is None:
                        smean = np.mean(series)
                    if fft is None:
                        nfft = int(
                            np.power(2, np.ceil(np.log(len(series)) / np.log(2)))
                        )
                        fft = np.fft.fft(series - smean, n=nfft)
                    if ac is None:
                        ac = _autocorr(series, fft)
                    args = [ac]
                elif feature == 16 or feature == 17 or feature == 20:
                    if smean is None:
                        smean = np.mean(series)
                    if fft is None:
                        nfft = int(
                            np.power(2, np.ceil(np.log(len(series)) / np.log(2)))
                        )
                        fft = np.fft.fft(series - smean, n=nfft)
                    if ac is None:
                        ac = _autocorr(series, fft)
                    if acfz is None:
                        acfz = _ac_first_zero(ac)
                    args = [series, acfz]

                if feature == 22:
                    c22[dim + n] = smean
                elif feature == 23:
                    c22[dim + n] = np.std(series)
                else:
                    c22[dim + n] = self._get_feature_fun(feature)(*args)

        return c22

    def _transform_single_feature(self, X, feature, case_id=None):
        if isinstance(feature, (int, np.integer)) or isinstance(
            feature, (float, float)
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
            X = convert_to(X, "numpy3D")

        if len(X.shape) > 2:
            n_instances, n_dims, series_length = X.shape

            if n_dims > 1:
                raise ValueError(
                    "transform_single_feature can only handle univariate series "
                    "currently."
                )

            X = np.reshape(X, (n_instances, -1))
        else:
            n_instances, series_length = X.shape

        if case_id is not None:
            if case_id != self._case_id:
                self._case_id = case_id
                self._st_n_instances = n_instances
                self._st_series_length = series_length
                self._outlier_series = [None] * n_instances
                self._smin = [None] * n_instances
                self._smax = [None] * n_instances
                self._smean = [None] * n_instances
                self._fft = [None] * n_instances
                self._ac = [None] * n_instances
                self._acfz = [None] * n_instances
            else:
                if (
                    n_instances != self._st_n_instances
                    or series_length != self._st_series_length
                ):
                    raise ValueError(
                        "Catch22: case_is the same, but n_instances and "
                        "series_length do not match last seen for single "
                        "feature transform."
                    )

        threads_to_use = check_n_jobs(self.n_jobs)

        if self.n_jobs == -1:
            threads_to_use = 1
            warnings.warn(
                "``n_jobs`` default was changed to 1 from -1 in version 0.13.4. "
                "In version 0.15 a value of -1 will use all CPU cores instead of the "
                "current 1 CPU core."
            )

        c22_list = Parallel(n_jobs=threads_to_use)(
            delayed(self._transform_case_single)(
                X[i],
                feature,
                case_id,
                i,
            )
            for i in range(n_instances)
        )

        if self.replace_nans:
            c22_list = np.nan_to_num(c22_list, False, 0, 0, 0)

        return np.asarray(c22_list)

    def _transform_case_single(self, series, feature, case_id, inst_idx):
        from sktime.transformations.panel._catch22_numba import (
            _ac_first_zero,
            _autocorr,
        )

        args = [series]

        if case_id is None:
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
        else:
            if feature == 0 or feature == 1 or feature == 11:
                if self._smin[inst_idx] is None:
                    self._smin[inst_idx] = np.min(series)
                if self._smax[inst_idx] is None:
                    self._smax[inst_idx] = np.max(series)
                args = [series, self._smin[inst_idx], self._smax[inst_idx]]
            elif feature == 2:
                if self._smean[inst_idx] is None:
                    self._smean[inst_idx] = np.mean(series)
                args = [series, self._smean[inst_idx]]
            elif feature == 3 or feature == 4:
                if self.outlier_norm:
                    if self._outlier_series[inst_idx] is None:
                        std = np.std(series)
                        if std > 0:
                            self._outlier_series[inst_idx] = (
                                series - np.mean(series)
                            ) / std
                        else:
                            self._outlier_series[inst_idx] = series
                    series = self._outlier_series[inst_idx]
                args = [series]
            elif feature == 7 or feature == 8:
                if self._smean[inst_idx] is None:
                    self._smean[inst_idx] = np.mean(series)
                if self._fft[inst_idx] is None:
                    nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
                    self._fft[inst_idx] = np.fft.fft(
                        series - self._smean[inst_idx], n=nfft
                    )
                args = [series, self._fft[inst_idx]]
            elif feature == 5 or feature == 6 or feature == 12:
                if self._smean[inst_idx] is None:
                    self._smean[inst_idx] = np.mean(series)
                if self._fft[inst_idx] is None:
                    nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
                    self._fft[inst_idx] = np.fft.fft(
                        series - self._smean[inst_idx], n=nfft
                    )
                if self._ac[inst_idx] is None:
                    self._ac[inst_idx] = _autocorr(series, self._fft[inst_idx])
                args = [self._ac[inst_idx]]
            elif feature == 16 or feature == 17 or feature == 20:
                if self._smean[inst_idx] is None:
                    self._smean[inst_idx] = np.mean(series)
                if self._fft[inst_idx] is None:
                    nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
                    self._fft[inst_idx] = np.fft.fft(
                        series - self._smean[inst_idx], n=nfft
                    )
                if self._ac[inst_idx] is None:
                    self._ac[inst_idx] = _autocorr(series, self._fft[inst_idx])
                if self._acfz[inst_idx] is None:
                    self._acfz[inst_idx] = _ac_first_zero(self._ac[inst_idx])
                args = [series, self._acfz[inst_idx]]

        return self._get_feature_fun(feature)(*args)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        param1 = {}
        param2 = {"features": "DN_HistogramMode_5"}
        return [param1, param2]
