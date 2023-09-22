"""Catch22 features.

A transformer for the Catch22 features using the pycatch22 C wrapper.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["Catch22Wrapper"]

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel import catch22
from sktime.utils.validation import check_n_jobs


class Catch22Wrapper(BaseTransformer):
    """Canonical Time-series Characteristics (Catch22) C Wrapper.

    Wraps the pycatch22 implementation for sktime
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
    n_jobs : int, optional, default=1
        The number of jobs to run in parallel for transform. Requires multiple input
        cases. ``-1`` means using all processors.

    See Also
    --------
    Catch22 Catch22Classifier

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
        "python_dependencies": "pycatch22",
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
        n_instances, n_dims = X.shape

        f_idx = catch22._verify_features(self.features, self.catch24)

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

        threads_to_use = check_n_jobs(self.n_jobs)

        c22_list = Parallel(n_jobs=threads_to_use)(
            delayed(self._transform_case)(
                X.iloc[i],
                f_idx,
                features,
            )
            for i in range(n_instances)
        )

        if self.replace_nans:
            c22_list = np.nan_to_num(c22_list, False, 0, 0, 0)

        return pd.DataFrame(c22_list)

    def _transform_case(self, X, f_idx, features):
        c22 = np.zeros(len(f_idx) * len(X))

        if self._transform_features is not None and len(
            self._transform_features
        ) == len(c22):
            transform_feature = self._transform_features
        else:
            transform_feature = [True] * len(c22)

        f_count = -1
        for i in range(len(X)):
            dim = i * len(f_idx)
            series = list(X[i])

            if self.outlier_norm and (3 in f_idx or 4 in f_idx):
                outlier_series = np.array(series)
                outlier_series = list(
                    catch22._normalise_series(outlier_series, np.mean(outlier_series))
                )

            for n, feature in enumerate(f_idx):
                f_count += 1
                if not transform_feature[f_count]:
                    continue

                if self.outlier_norm and feature in [3, 4]:
                    c22[dim + n] = features[feature](outlier_series)
                if feature == 22:
                    c22[dim + n] = np.mean(series)
                elif feature == 23:
                    c22[dim + n] = np.std(series)
                else:
                    c22[dim + n] = features[feature](series)

        return c22

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
        param2 = {"features": [1, 3, 7]}
        return [param1, param2]


feature_names = catch22.feature_names
