# -*- coding: utf-8 -*-
""" catch22 features
A transformer for the catch22 features
"""

__author__ = "Matthew Middlehurst"
__all__ = ["Catch22"]

import numpy as np
import pandas as pd
from sktime.transformers.base import _PanelToTabularTransformer
from sktime.utils.check_imports import _check_soft_dependencies
from sktime.utils.data_container import from_nested_to_2d_array
from sktime.utils.validation.panel import check_X

_check_soft_dependencies("catch22")
import catch22  # noqa: E402


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

    Overview: Input n series length m
    Transforms series into the 22 catch22 features extracted from the hctsa
    toolbox.

    Fulcher, B. D., & Jones, N. S. (2017). hctsa: A computational framework
    for automated time-series phenotyping using massive feature extraction.
    Cell systems, 5(5), 527-531.

    Fulcher, B. D., Little, M. A., & Jones, N. S. (2013). Highly comparative
    time-series analysis: the empirical structure of time series and their
    methods. Journal of the Royal Society Interface, 10(83), 20130048.

    catch22 package implementations:
    https://github.com/chlubba/catch22

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/transformers/Catch22.java

    """

    def __init__(self):
        super(Catch22, self).__init__()

    def transform(self, X, y=None):
        """transforms data into the catch22 features

        Parameters
        ----------
        X : pandas DataFrame, input time series
        y : array_like, target values (optional, ignored)

        Returns
        -------
        Pandas dataframe containing 22 features for each input series
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)
        n_instances = X.shape[0]
        X = np.reshape(X, (n_instances, -1))

        c22_list = []
        for i in range(n_instances):
            series = X[i, :]
            c22_dict = catch22.catch22_all(series)
            c22_list.append(c22_dict["values"])

        return pd.DataFrame(c22_list)

    def _transform_single_feature(self, X, feature):
        """transforms data into the catch22 features

        Parameters
        ----------
        X : pandas DataFrame, input time series
        feature : int, catch22 feature id or String, catch22 feature
                  name.

        Returns
        -------
        Numpy array containing a catch22 feature for each input series
        """
        if isinstance(feature, int):
            if feature > 21 or feature < 0:
                raise ValueError("Invalid catch22 feature ID")
        elif isinstance(feature, str):
            if feature in feature_names:
                feature = feature_names.index(feature)
            else:
                raise ValueError("Invalid catch22 feature name")
        else:
            raise ValueError("Feature name or ID required")

        if isinstance(X, pd.DataFrame):
            X = from_nested_to_2d_array(X, return_numpy=True)

        n_instances = X.shape[0]
        X = np.reshape(X, (n_instances, -1))

        c22_list = []
        for i in range(n_instances):
            series = X[i, :].tolist()
            c22_val = features[feature](series)
            c22_list.append(c22_val)

        return np.array(c22_list)


feature_names = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]

features = [
    catch22.DN_HistogramMode_5,
    catch22.DN_HistogramMode_10,
    catch22.CO_f1ecac,
    catch22.CO_FirstMin_ac,
    catch22.CO_HistogramAMI_even_2_5,
    catch22.CO_trev_1_num,
    catch22.MD_hrv_classic_pnn40,
    catch22.SB_BinaryStats_mean_longstretch1,
    catch22.SB_TransitionMatrix_3ac_sumdiagcov,
    catch22.PD_PeriodicityWang_th0_01,
    catch22.CO_Embed2_Dist_tau_d_expfit_meandiff,
    catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi,
    catch22.FC_LocalSimple_mean1_tauresrat,
    catch22.DN_OutlierInclude_p_001_mdrmd,
    catch22.DN_OutlierInclude_n_001_mdrmd,
    catch22.SP_Summaries_welch_rect_area_5_1,
    catch22.SB_BinaryStats_diff_longstretch0,
    catch22.SB_MotifThree_quantile_hh,
    catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    catch22.SP_Summaries_welch_rect_centroid,
    catch22.FC_LocalSimple_mean3_stderr,
]
