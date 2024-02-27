"""Catch22 features.

A transformer for the Catch22 features.
"""

__author__ = ["MatthewMiddlehurst", "julnow"]
__all__ = ["Catch22"]

from typing import List, Union

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel._catch22_numba import (
    _ac_first_zero,
    _autocorr,
    _catch24_mean,
    _catch24_std,
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
    _normalise_series,
    _PD_PeriodicityWang_th0_01,
    _SB_BinaryStats_diff_longstretch0,
    _SB_BinaryStats_mean_longstretch1,
    _SB_MotifThree_quantile_hh,
    _SB_TransitionMatrix_3ac_sumdiagcov,
    _SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    _SC_FluctAnal_2_rsrangefit_50_1_logi,
    _SP_Summaries_welch_rect_area_5_1,
    _SP_Summaries_welch_rect_centroid,
)
from sktime.utils.warnings import warn

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
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": _SC_FluctAnal_2_rsrangefit_50_1_logi,
    "SB_TransitionMatrix_3ac_sumdiagcov": _SB_TransitionMatrix_3ac_sumdiagcov,
    "PD_PeriodicityWang_th0_01": _PD_PeriodicityWang_th0_01,
}
FEATURE_NAMES = list(METHODS_DICT.keys())
SHORT_FEATURE_NAMES_DICT = {
    "DN_HistogramMode_5": "mode_5",
    "DN_HistogramMode_10": "mode_10",
    "SB_BinaryStats_diff_longstretch0": "stretch_decreasing",
    "DN_OutlierInclude_p_001_mdrmd": "outlier_timing_pos",
    "DN_OutlierInclude_n_001_mdrmd": "outlier_timing_neg",
    "CO_f1ecac": "acf_timescale",
    "CO_FirstMin_ac": "acf_first_min",
    "SP_Summaries_welch_rect_area_5_1": "centroid_freq",
    "SP_Summaries_welch_rect_centroid": "low_freq_power",
    "FC_LocalSimple_mean3_stderr": "forecast_error",
    "CO_trev_1_num": "trev",
    "CO_HistogramAMI_even_2_5": "ami2",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": "ami_timescale",
    "MD_hrv_classic_pnn40": "high_fluctuation",
    "SB_BinaryStats_mean_longstretch1": "stretch_high",
    "SB_MotifThree_quantile_hh": "rs_range",
    "FC_LocalSimple_mean1_tauresrat": "whiten_timescale",
    "CO_Embed2_Dist_tau_d_expfit_meandiff": "embedding_dist",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": "dfa",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": "rs_range",
    "SB_TransitionMatrix_3ac_sumdiagcov": "transition_matrix",
    "PD_PeriodicityWang_th0_01": "periodicity",
}
SHORT_FEATURE_NAMES = list(SHORT_FEATURE_NAMES_DICT.values())

CATCH24_METHODS_DICT = {"Mean": _catch24_mean, "StandardDeviation": _catch24_std}
CATCH24_FEATURE_NAMES = list(CATCH24_METHODS_DICT.keys())
CATCH24_SHORT_FEATURE_NAMES_DICT = {
    "Mean": "mean",
    "StandardDeviation": "std",
}
CATCH24_SHORT_FEATURE_NAMES = list(CATCH24_SHORT_FEATURE_NAMES_DICT.values())


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
    col_names : str, one of {"range", "int_feat", "str_feat", "short_str_feat"},
    optional, default="range"
        The type of column names to return. If "range", column names will be
        a regular range of integers, as in a RangeIndex.
        If "int_feat", column names will be the integer feature indices,
        as defined in pycatch22.
        If "str_feat", column names will be the string feature names.
        If "short_str_feat", column names will be the short string feature names
        as defined in pycatch22.

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
        "univariate-only": True,
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.Series",
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
        col_names: str = "range",
        n_jobs="deprecated",
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.col_names = self._set_col_names(col_names)
        self.f_idx = _verify_features(self.features, self.catch24)
        super().__init__()

        self.n_jobs = n_jobs

        # todo 0.28.0: remove this warning and logic
        if n_jobs != "deprecated":
            warn(
                "In Catch22Wrapper, the parameter "
                "n_jobs is deprecated and will be removed in v0.28.0. "
                "Instead, use set_config with the backend and backend:params "
                "config fields, and set backend to 'joblib' and pass n_jobs "
                "as a parameter of backend_params. ",
                FutureWarning,
                obj=self,
            )
            self.set_config(backend="joblib", backend_params={"n_jobs": n_jobs})

    def _set_col_names(self, col_names: str) -> str:
        """Set valid column names type.

        Check and return col_names if one of:
        ["range", "int_feat", "str_feat", "short_str_feat"].


        Parameters
        ----------
        col_names : str with type of desired col_names

        Returns
        -------
        col_names string which should be one of acceptable types.

        Raises
        ------
        KeyError if not in accepted col_names types.
        """
        accepted_col_names = ["range", "int_feat", "str_feat", "short_str_feat"]
        if col_names in accepted_col_names:
            return col_names
        else:
            raise KeyError(
                f"col_names type: {col_names} must be one of {accepted_col_names}"
            )

    def _transform(self, X: pd.Series, y=None) -> pd.DataFrame:
        """Transform data into the Catch22 features.

        Parameters
        ----------
        X : pd.Series with input univariate time series panel.
        y : ignored.

        Returns
        -------
        c22 : Pandas DataFrame of shape [n_instances, c*n_dimensions] where c is the
             number of features requested, containing Catch22 features for X.
        """
        Xt = self._transform_case(X, self.f_idx)
        if self.replace_nans:
            Xt = Xt.fillna(0)

        return Xt

    def _get_feature_function(self, feature: Union[int, str]):
        if isinstance(feature, int):
            return (
                METHODS_DICT.get(FEATURE_NAMES[feature])
                if feature < 22
                else CATCH24_METHODS_DICT.get(CATCH24_FEATURE_NAMES[feature - 22])
            )
        elif isinstance(feature, str):
            if feature in FEATURE_NAMES:
                return METHODS_DICT.get(feature)
            if feature in CATCH24_FEATURE_NAMES:
                return CATCH24_METHODS_DICT.get(feature)
        else:
            raise KeyError(f"No feature with name: {feature}")

    def _transform_case(self, X: pd.Series, f_idx: List[int]) -> pd.DataFrame:
        """Transform data into the Catch22/24 features.

        Parameters
        ----------
        X : pd.Series, input time series.
        f_idx : list of int, the indices of the features to extract.

        Returns
        -------
        Xt : pd.DataFrame of size [1, n_features], where n_features is the
            number of features requested, containing Catch22/24 features for X.
            column index is determined by self.col_names
        """
        n_features = len(f_idx)
        Xt_np = np.zeros((1, n_features))

        series = X.to_list()
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
        variable_dict = {
            "series": series,
            "smin": smin,
            "smax": smax,
            "smean": smean,
            "std": std,
            "outlier_series": outlier_series,
            "fft": fft,
            "ac": ac,
            "acfz": acfz,
        }

        for n, feature in enumerate(f_idx):
            Xt_np[0, n] = self._get_feature_function(feature)(variable_dict)

        cols = self._prepare_output_col_names(n_features)

        return pd.DataFrame(Xt_np, columns=cols)

    def _prepare_output_col_names(
        self, n_features: int
    ) -> Union[range, List[int], List[str]]:
        """Prepare output column names.

        It selects the naming style according to self.col_names.
        If "int_feat", column names will be the integer feature indices,
        as defined in pycatch22.
        If "str_feat", column names will be the string feature names.
        If "short_str_feat", column names will be the short string feature names
        as defined in pycatch22.

        Parameters
        ----------
        n_features : int
            Number of features in f_idx.

        Returns
        -------
        Union[range, List[int], List[str]]
            Column labels for ouput DataFrame.
        """
        if self.col_names == "range":
            return range(n_features)
        elif self.col_names == "int_feat":
            return self.f_idx
        elif self.col_names == "str_feat":
            all_feature_names = (
                FEATURE_NAMES + CATCH24_FEATURE_NAMES if self.catch24 else FEATURE_NAMES
            )
            return [all_feature_names[i] for i in self.f_idx]
        elif self.col_names == "short_str_feat":
            all_short_feature_names = (
                SHORT_FEATURE_NAMES + CATCH24_SHORT_FEATURE_NAMES
                if self.catch24
                else SHORT_FEATURE_NAMES
            )
            return [all_short_feature_names[i] for i in self.f_idx]

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
