"""Catch22 features.

A transformer for the Catch22 features.
"""

__author__ = ["MatthewMiddlehurst", "julnow"]
__all__ = ["Catch22"]

from typing import Union

import numpy as np
import pandas as pd

from sktime.datatypes import convert_to
from sktime.transformations.base import BaseTransformer
from sktime.utils.warnings import warn


def get_methods_dict(which="catch22"):
    from sktime.transformations.panel._catch22_numba import (
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

    CATCH22_METHODS_DICT = {
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
        "IN_AutoMutualInfoStats_40_gaussian_fmmi": _IN_AutoMutualInfoStats_40_gaussian_fmmi,  # noqa: E501
        "MD_hrv_classic_pnn40": _MD_hrv_classic_pnn40,
        "SB_BinaryStats_mean_longstretch1": _SB_BinaryStats_mean_longstretch1,
        "SB_MotifThree_quantile_hh": _SB_MotifThree_quantile_hh,
        "FC_LocalSimple_mean1_tauresrat": _FC_LocalSimple_mean1_tauresrat,
        "CO_Embed2_Dist_tau_d_expfit_meandiff": _CO_Embed2_Dist_tau_d_expfit_meandiff,
        "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": _SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,  # noqa: E501
        "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": _SC_FluctAnal_2_rsrangefit_50_1_logi,  # noqa: E501
        "SB_TransitionMatrix_3ac_sumdiagcov": _SB_TransitionMatrix_3ac_sumdiagcov,
        "PD_PeriodicityWang_th0_01": _PD_PeriodicityWang_th0_01,
    }
    CATCH24_METHODS_DICT = {"DN_Mean": _catch24_mean, "DN_Spread_Std": _catch24_std}

    if which == "catch22":
        return CATCH22_METHODS_DICT
    elif which == "catch24":
        return CATCH24_METHODS_DICT
    else:
        raise ValueError(
            "Invalid value for parameter 'which' in catch22.get_methods_dict, "
            "must be one of the strings 'catch22' or 'catch24'."
        )


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
    "SB_MotifThree_quantile_hh": "entropy_pairs",
    "FC_LocalSimple_mean1_tauresrat": "whiten_timescale",
    "CO_Embed2_Dist_tau_d_expfit_meandiff": "embedding_dist",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": "dfa",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": "rs_range",
    "SB_TransitionMatrix_3ac_sumdiagcov": "transition_matrix",
    "PD_PeriodicityWang_th0_01": "periodicity",
}
SHORT_FEATURE_NAMES = list(SHORT_FEATURE_NAMES_DICT.values())
FEATURE_NAMES = list(SHORT_FEATURE_NAMES_DICT.keys())
feature_names = FEATURE_NAMES  # for backwards compatibility, this variable was public

CATCH24_SHORT_FEATURE_NAMES_DICT = {
    "DN_Mean": "mean",
    "DN_Spread_Std": "std",
}
CATCH24_SHORT_FEATURE_NAMES = list(CATCH24_SHORT_FEATURE_NAMES_DICT.values())
CATCH24_FEATURE_NAMES = list(CATCH24_SHORT_FEATURE_NAMES_DICT.keys())


def _verify_features(
    features: Union[int, str, list[Union[int, str]]], catch24: bool
) -> list[int]:
    feature_names = FEATURE_NAMES + CATCH24_FEATURE_NAMES if catch24 else FEATURE_NAMES
    short_feature_names = (
        SHORT_FEATURE_NAMES + CATCH24_SHORT_FEATURE_NAMES
        if catch24
        else SHORT_FEATURE_NAMES
    )
    f_idx = []
    if isinstance(features, str):
        if features == "all":
            f_idx = list(range(len(feature_names)))
        elif features in feature_names:
            f_idx = [feature_names.index(features)]
        elif features in short_feature_names:
            f_idx = [short_feature_names.index(features)]
    elif isinstance(features, int) and 0 <= features < (24 if catch24 else 22):
        f_idx = [features]
    elif isinstance(features, (list, tuple)):
        for f in features:
            if isinstance(f, str):
                if f in feature_names:
                    f_idx.append(feature_names.index(f))
                elif f in short_feature_names:
                    f_idx.append(short_feature_names.index(f))

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
        Valid features and their corresponding short feature names are as follows:
            {
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
            "SB_MotifThree_quantile_hh": "entropy_pairs",
            "FC_LocalSimple_mean1_tauresrat": "whiten_timescale",
            "CO_Embed2_Dist_tau_d_expfit_meandiff": "embedding_dist",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": "dfa",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": "rs_range",
            "SB_TransitionMatrix_3ac_sumdiagcov": "transition_matrix",
            "PD_PeriodicityWang_th0_01": "periodicity",
        }
        Additionally, if catch24 is true, two additional features are available:
        {
            "DN_Mean": "mean",
            "DN_Spread_Std": "std",
        }
        The overview of these features is available at:
        https://time-series-features.gitbook.io/catch22-features/feature-overview-table
    catch24 : bool, optional, default=False
        Extract the mean and standard deviation as well as the 22 Catch22 features if
        true. If a List of specific features to extract is provided, "Mean" and/or
        "StandardDeviation" must be added to the List to extract these features.
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features:
        `_DN_OutlierInclude_p_001_mdrmd` and `_DN_OutlierInclude_n_001_mdrmd`,
        which can take a while to process for large values.
    replace_nans : bool, optional, default=True
        Replace NaN or inf values from the Catch22 transform with 0.
    col_names : str, one of {"range", "int_feat", "str_feat", "short_str_feat", "auto"},
    optional, default="range"
        The type of column names to return. If "range", column names will be
        a regular range of integers, as in a RangeIndex.
        If "int_feat", column names will be the integer feature indices,
        as defined in pycatch22.
        If "str_feat", column names will be the string feature names.
        If "short_str_feat", column names will be the short string feature names
        as defined in pycatch22.
        If "auto", column names will be the same as defined in features.

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
        # packaging info
        # --------------
        "authors": ["julnow", "MatthewMiddlehurst"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "univariate-only": True,
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
    }

    def __init__(
        self,
        features: Union[int, str, list[Union[int, str]]] = "all",
        catch24: bool = False,
        outlier_norm: bool = False,
        replace_nans: bool = False,
        col_names: str = "range",
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.col_names = self._set_col_names(col_names)
        self.f_idx = _verify_features(self.features, self.catch24)

        # todo: remove this unimplemented logic
        self._transform_features = None

        super().__init__()

        self.METHODS_DICT = get_methods_dict("catch22")
        self.CATCH24_METHODS_DICT = get_methods_dict("catch24")

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
        accepted_col_names = ["range", "int_feat", "str_feat", "short_str_feat", "auto"]
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
            column index is determined by self.col_names
        """
        Xt_np = self._transform_case(X, self.f_idx)
        cols = self._prepare_output_col_names(len(self.f_idx))

        Xt = pd.DataFrame(Xt_np, columns=cols)
        if self.replace_nans:
            Xt = Xt.fillna(0)
        return Xt

    # todo: remove case_id
    def _transform_single_feature(
        self, X: pd.Series, feature: Union[int, str], case_id="deprecated"
    ):
        if case_id != "deprecated":
            warn(
                "In Catch22._transform_single_feature, the argument "
                "case_id is deprecated and will be removed in the future.",
                FutureWarning,
                obj=self,
            )
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

        # todo: remove Parallel in future versions, left for
        # compatibility with `CanonicalIntervalForest`
        c22_list = [self._transform_case(X[i], [feature]) for i in range(n_instances)]

        if self.replace_nans:
            c22_list = np.nan_to_num(c22_list, False, 0, 0, 0)

        return np.asarray(c22_list)[:, 0, 0]

    def _transform_case(self, X: pd.Series, f_idx: list[int]) -> np.ndarray:
        """Transform data into the Catch22/24 features.

        Parameters
        ----------
        X : pd.Series, input time series.
        f_idx : list of int, the indices of the features to extract.

        Returns
        -------
        Xt : np.ndarray of size [1, n_features], where n_features is the
            number of features requested, containing Catch22/24 features for X.
        """
        from sktime.transformations.panel._catch22_numba import (
            _ac_first_zero,
            _autocorr,
            _normalise_series,
        )

        n_features = len(f_idx)
        Xt_np = np.zeros((1, n_features))

        series = X if isinstance(X, np.ndarray) else X.to_numpy()
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
        # todo: remove unimplemented logic
        if (
            self._transform_features is not None
            and len(self._transform_features) == n_features
        ):
            transform_feature = self._transform_features
        else:
            transform_feature = [True] * n_features

        for n, feature in enumerate(f_idx):
            # todo: remove unimplemented logic
            if not transform_feature[n]:
                continue
            Xt_np[0, n] = (
                self._get_feature_function(feature)(
                    series, smin, smax, smean, std, outlier_series, ac, acfz
                )
                or None
            )

        return Xt_np

    def _get_feature_function(self, feature: Union[int, str]):
        if isinstance(feature, str):
            return self.__get_feature_function_str(feature)
        return self.__get_feature_function_int(feature)

    def __get_feature_function_str(self, feature: str):
        if feature in FEATURE_NAMES:
            return self.METHODS_DICT.get(feature)
        if feature in CATCH24_FEATURE_NAMES:
            return self.CATCH24_METHODS_DICT.get(feature)
        else:
            return self.__get_feature_function_int(int(str))

    def __get_feature_function_int(self, feature: int):
        if feature < 22:
            return self.METHODS_DICT.get(FEATURE_NAMES[feature])
        if 22 <= feature < 24:
            return self.CATCH24_METHODS_DICT.get(CATCH24_FEATURE_NAMES[feature - 22])
        else:
            raise KeyError(f"No feature with name: {feature}")

    def _prepare_output_col_names(
        self, n_features: int
    ) -> Union[range, list[int], list[str]]:
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
            Column labels for output DataFrame.
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
        elif self.col_names == "auto":
            return self.f_idx

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        param1 = {}
        param2 = {"features": "DN_HistogramMode_5", "col_names": "int_feat"}
        param3 = {
            "features": [1, 12, 23],
            "catch24": True,
            "replace_nans": True,
            "col_names": "str_feat",
        }
        param4 = {
            "features": ["forecast_error", "_DN_OutlierInclude_p_001_mdrmd"],
            "outlier_norm": True,
            "col_names": "short_str_feat",
        }
        param5 = {
            "features": [11, "DN_HistogramMode_5", "forecast_error"],
            "col_names": "auto",
        }
        return [param1, param2, param3, param4, param5]
