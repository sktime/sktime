"""Catch22 features.

A transformer for the Catch22 features.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["Catch22"]

from typing import List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sktime.datatypes import convert_to
from sktime.transformations.base import BaseTransformer
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
from sktime.utils.validation import check_n_jobs

METHODS_DICT = {
    "DN_HistogramMode_5": _DN_HistogramMode_5,
    "DN_HistogramMode_10": _DN_HistogramMode_10,
    "SB_BinaryStats_diff_longstretch0": _SB_BinaryStats_diff_longstretch0,
    "DN_OutlierInclude_p_001_mdrmd": _DN_OutlierInclude_p_001_mdrmd,
    "DN_OutlierInclude_n_001_mdrmd": _DN_OutlierInclude_n_001_mdrmd,
    "CO_f1ecac": _CO_f1ecac,
    "CO_FirstMin_ac": _CO_FirstMin_ac,
    "SP_Summaries_welch_rect_area_5_1": _SP_Summaries_welch_rect_area_5_1,
    "SP_Summaries_welch_rect_centroid": _SP_Summaries_welch_rect_centroid,
    "FC_LocalSimple_mean3_stderr": _FC_LocalSimple_mean3_stderr,
    "CO_trev_1_num": _CO_trev_1_num,
    "CO_HistogramAMI_even_2_5": _CO_HistogramAMI_even_2_5,
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": _IN_AutoMutualInfoStats_40_gaussian_fmmi,
    "MD_hrv_classic_pnn40": _MD_hrv_classic_pnn40,
    "SB_BinaryStats_mean_longstretch1": _SB_BinaryStats_mean_longstretch1,
    "SB_MotifThree_quantile_hh": _SB_MotifThree_quantile_hh,
    "FC_LocalSimple_mean1_tauresrat": _FC_LocalSimple_mean1_tauresrat,
    "CO_Embed2_Dist_tau_d_expfit_meandiff": _CO_Embed2_Dist_tau_d_expfit_meandiff,
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": _SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    "SC_FluctAnal_2_rsrangefit_50_1_logi": _SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    "SB_TransitionMatrix_3ac_sumdiagcov": _SB_TransitionMatrix_3ac_sumdiagcov,
    "PD_PeriodicityWang_th0_01": _PD_PeriodicityWang_th0_01,
}

FEATURE_NAMES = list(METHODS_DICT.keys())

CATCH24_FEATURE_NAMES = ["Mean", "StandardDeviation"]


def _verify_features(
    features: Union[int, str, List[Union[int, str]]], catch24: bool
) -> List[int]:
    feature_names = FEATURE_NAMES + CATCH24_FEATURE_NAMES if catch24 else FEATURE_NAMES

    f_idx = []
    if isinstance(features, str):
        if features == "all":
            f_idx = list(range(len(feature_names)))
        elif features in feature_names:
            f_idx = [feature_names.index(features)]
    elif isinstance(features, int) and 0 <= features < (24 if catch24 else 22):
        f_idx = [features]
    elif isinstance(features, (list, tuple)):
        for f in features:
            if isinstance(f, str) and f in feature_names:
                f_idx.append(feature_names.index(f))
            elif isinstance(f, int) and 0 <= f < (24 if catch24 else 22):
                f_idx.append(f)

    if not f_idx:
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
        features: Union[int, str, List[Union[int, str]]] = "all",
        catch24: bool = False,
        outlier_norm: bool = False,
        replace_nans: bool = False,
        n_jobs: int = 1,
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.n_jobs = n_jobs

        self.features_arguments = (
            features
            if features != "all"
            else (FEATURE_NAMES + CATCH24_FEATURE_NAMES if catch24 else FEATURE_NAMES)
        )
        self.f_idx = _verify_features(self.features, self.catch24)
        self.n_transformed_features = len(self.f_idx)

        super().__init__()

    def _transform(self, X: np.ndarray, y=None):
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

        threads_to_use = check_n_jobs(self.n_jobs)

        c22_list = Parallel(n_jobs=threads_to_use)(
            delayed(self._transform_case)(
                X.iloc[i],
                self.f_idx,
            )
            for i in range(n_instances)
        )

        if self.replace_nans:
            c22_list = np.nan_to_num(c22_list, False, 0, 0, 0)

        return pd.DataFrame(c22_list)

    def _get_feature_function(self, feature: Union[int, str]):
        feature_name = FEATURE_NAMES[feature] if isinstance(feature, int) else feature

        return METHODS_DICT.get(feature_name)

    def _transform_case(self, X: np.ndarray, f_idx: List[int]):
        from sktime.transformations.panel._catch22_numba import (
            _ac_first_zero,
            _autocorr,
            _normalise_series,
        )

        c22 = np.zeros(len(f_idx) * len(X))

        for i, series in enumerate(X):
            series = np.array(series)
            dim = i * len(f_idx)
            smin = np.min(series)
            smax = np.max(series)
            smean = np.mean(series)
            std = np.std(series)
            outlier_series = (
                _normalise_series(series, smean) if self.outlier_norm else series
            )
            nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
            fft = np.fft.fft(series - smean, n=nfft)
            ac = _autocorr(series, fft)
            acfz = _ac_first_zero(ac)
            feature_args = {
                0: (series, smin, smax),
                1: (series, smin, smax),
                11: (series, smin, smax),
                2: (series, smean),
                22: (series, smean),
                3: (outlier_series,),
                4: (outlier_series,),
                7: (series, fft),
                8: (series, fft),
                5: (ac,),
                6: (ac,),
                12: (ac,),
                16: (series, acfz),
                17: (series, acfz),
                20: (series, acfz),
            }

            for n, feature in enumerate(f_idx):
                args = feature_args.get(feature, (series,))

                if feature == 22:
                    c22[dim + n] = smean
                elif feature == 23:
                    c22[dim + n] = std
                else:
                    c22[dim + n] = self._get_feature_function(feature)(*args)

            return c22

    def _transform_single_feature(self, X: np.ndarray, feature: Union[int, str]):
        if isinstance(X, pd.DataFrame):
            X = convert_to(X, "numpy3D")

        if len(X.shape) > 2:
            n_instances, n_dims, _ = X.shape

            if n_dims > 1:
                raise ValueError(
                    "transform_single_feature can only handle univariate series "
                    "currently."
                )

            X = np.reshape(X, (n_instances, -1))
        else:
            n_instances, _ = X.shape

        threads_to_use = check_n_jobs(self.n_jobs)

        c22_list = Parallel(n_jobs=threads_to_use)(
            delayed(self._transform_case_single)(
                X.iloc[i],
                feature,
            )
            for i in range(n_instances)
        )

        if self.replace_nans:
            c22_list = np.nan_to_num(c22_list, False, 0, 0, 0)

        return np.asarray(c22_list)

    def _transform_case_single(self, series, feature):
        from sktime.transformations.panel._catch22_numba import (
            _ac_first_zero,
            _autocorr,
        )

        smin = np.min(series)
        smax = np.max(series)
        smean = np.mean(series)
        if self.outlier_norm:
            std = np.std(series)
            if std > 0:
                series = (series - np.mean(series)) / std
        nfft = int(np.power(2, np.ceil(np.log(len(series)) / np.log(2))))
        fft = np.fft.fft(series - smean, n=nfft)
        ac = _autocorr(series, fft)
        acfz = _ac_first_zero(ac)
        args = [series]

        feature_args = {
            0: (series, smin, smax),
            1: (series, smin, smax),
            11: (series, smin, smax),
            2: (series, smean),
            3: (series,),
            4: (series,),
            7: (series, fft),
            8: (series, fft),
            5: (ac,),
            6: (ac,),
            12: (ac,),
            16: (series, acfz),
            17: (series, acfz),
            20: (series, acfz),
        }
        args = feature_args.get(feature, (series,))

        return self._get_feature_function(feature)(*args)

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
