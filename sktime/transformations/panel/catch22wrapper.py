"""Catch22 features.

A transformer for the Catch22 features using the pycatch22 C wrapper.
"""

__author__ = ["MatthewMiddlehurst", "fkiraly"]
__all__ = ["Catch22Wrapper"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel import catch22


class Catch22Wrapper(BaseTransformer):
    """Canonical Time-series Characteristics (Catch22 and 24), using pycatch22 package.

    Direct interface to the ``pycatch22`` implementation of Catch-22 and Catch-24
    feature sets
    (https://github.com/DynamicsAndNeuralSystems/pycatch22).

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
    col_names : str, one of {"range", "int_feat", "str_feat"}, optional, default="range"
        The type of column names to return. If "range", column names will be
        a regular range of integers, as in a RangeIndex.
        If "int_feat", column names will be the integer feature indices,
        as defined in pycatch22.
        If "str_feat", column names will be the string feature names.

    See Also
    --------
    Catch22
    Catch22Classifier

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
        "authors": ["benfulcher", "jmoo2880", "MatthewMiddlehurst", "fkiraly"],
        "maintainers": ["benfulcher", "jmoo2880"],
        "python_dependencies": "pycatch22",
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "univariate-only": True,
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        features="all",
        catch24=False,
        outlier_norm=False,
        replace_nans=False,
        col_names="range",
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.col_names = col_names

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

        super().__init__()

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
        f_idx = catch22._verify_features(self.features, self.catch24)

        Xt = self._transform_case(X, f_idx)

        if self.replace_nans:
            Xt = Xt.fillna(0)

        return Xt

    def _get_fun_with_ix(self, ix):
        """Return function with index ix from the pycatch22 library.

        Index 0 - 21 are the catch22 features, in the same order as in pycatch22.
        22 is the mean and 23 is the standard deviation.
        If self.outlier_norm is True, features 3 and 4 have outlier normalisation
        applied to them before calculation, also from pycatch22.

        Parameters
        ----------
        ix : int, index of the function to return.

        Returns
        -------
        A function with the signature f(series: List[float]) -> float
        """
        import pycatch22

        features = [
            pycatch22.DN_HistogramMode_5,
            pycatch22.DN_HistogramMode_10,
            pycatch22.SB_BinaryStats_diff_longstretch0,
            pycatch22.DN_OutlierInclude_p_001_mdrmd,
            pycatch22.DN_OutlierInclude_n_001_mdrmd,
            pycatch22.CO_f1ecac,
            pycatch22.CO_FirstMin_ac,
            pycatch22.SP_Summaries_welch_rect_area_5_1,
            pycatch22.SP_Summaries_welch_rect_centroid,
            pycatch22.FC_LocalSimple_mean3_stderr,
            pycatch22.CO_trev_1_num,
            pycatch22.CO_HistogramAMI_even_2_5,
            pycatch22.IN_AutoMutualInfoStats_40_gaussian_fmmi,
            pycatch22.MD_hrv_classic_pnn40,
            pycatch22.SB_BinaryStats_mean_longstretch1,
            pycatch22.SB_MotifThree_quantile_hh,
            pycatch22.FC_LocalSimple_mean1_tauresrat,
            pycatch22.CO_Embed2_Dist_tau_d_expfit_meandiff,
            pycatch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
            pycatch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
            pycatch22.SB_TransitionMatrix_3ac_sumdiagcov,
            pycatch22.PD_PeriodicityWang_th0_01,
        ]

        def _feature_with_outlier_norm(X):
            X = _normalise_series(X)
            return features[ix](X)

        if ix in [3, 4]:
            return _feature_with_outlier_norm
        elif ix == 22:
            return np.mean
        elif ix == 23:
            return np.std
        else:
            return features[ix]

    def _transform_case(self, X, f_idx):
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
        n_feat = len(f_idx)
        Xt_np = np.zeros((1, n_feat))

        if (
            self._transform_features is not None
            and len(self._transform_features) == n_feat
        ):
            transform_feature = self._transform_features
        else:
            transform_feature = [True] * n_feat

        f_count = -1

        series = X.to_list()

        for n, feature in enumerate(f_idx):
            f_count += 1
            if not transform_feature[f_count]:
                continue

            feat_fun = self._get_fun_with_ix(feature)
            Xt_np[0, n] = feat_fun(series)

        col_names = self.col_names

        if col_names == "range":
            cols = range(n_feat)
        elif col_names == "int_feat":
            cols = f_idx
        elif col_names == "str_feat":
            all_feature_names = feature_names + ["Mean", "StandardDeviation"]
            cols = [all_feature_names[i] for i in f_idx]

        Xt = pd.DataFrame(Xt_np, columns=cols)

        return Xt

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
        param1 = {"catch24": True}
        param2 = {"features": [1, 3, 7], "col_names": "int_feat"}
        param3 = {"features": "all", "outlier_norm": True, "replace_nans": True}
        param4 = {"features": ["CO_trev_1_num"], "col_names": "str_feat"}
        return [param1, param2, param3, param4]


feature_names = catch22.FEATURE_NAMES


def _normalise_series(X):
    X = np.array(X)
    std = np.std(X)
    mean = np.mean(X)
    if std > 0:
        X_norm = (X - mean) / std
    else:
        X_norm = X
    return list(X_norm)
