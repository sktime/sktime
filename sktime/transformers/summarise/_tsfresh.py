#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Ayushmaan Seth", "Markus Löning"]
__all__ = ["TSFreshFeatureExtractor", "TSFreshRelevantFeatureExtractor"]

import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sktime.transformers.base import BaseTransformer
from sktime.utils.data_container import from_nested_to_long
from sktime.utils.validation.supervised import validate_X
from sktime.utils.validation.supervised import validate_y


class BaseTSFreshFeatureExtractor(BaseTransformer):
    """Base class for sktime Transformers"""

    def __init__(self, default_fc_parameters="comprehensive", kind_to_fc_parameters=None, chunksize=None,
                 n_jobs=None, show_warnings=None, disable_progressbar=None,
                 impute_function=None, profiling=None, profiling_filename=None,
                 profiling_sorting=None, distributor=None, keep_time_series=False):
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.impute_function = impute_function
        self.profiling = profiling
        self.profiling_sorting = profiling_sorting
        self.profiling_filename = profiling_filename
        self.distributor = distributor
        self.keep_original = keep_time_series

    def fit(self, X, y=None):
        """Fit.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        y : pd.Series or np.array
            Target variable

        Returns
        -------
        self : an instance of self
        """
        validate_X(X)
        self._set_extraction_defaults()
        self._is_fitted = True
        return self

    def _set_extraction_defaults(self):
        """Helper function to set default parameters for feature extraction

        Dependencies: tsfresh

        tsfresh defaults and utilities used for parameters
        """
        from tsfresh.defaults import CHUNKSIZE
        from tsfresh.defaults import DISABLE_PROGRESSBAR
        from tsfresh.utilities.dataframe_functions import impute
        from tsfresh.defaults import N_PROCESSES
        from tsfresh.defaults import PROFILING
        from tsfresh.defaults import PROFILING_FILENAME
        from tsfresh.defaults import PROFILING_SORTING
        from tsfresh.defaults import SHOW_WARNINGS
        from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
        from tsfresh.feature_extraction.settings import EfficientFCParameters
        from tsfresh.feature_extraction.settings import MinimalFCParameters

        defaults = {
            "n_jobs": N_PROCESSES,
            "chunksize": CHUNKSIZE,
            "show_warnings": SHOW_WARNINGS,
            "disable_progressbar": DISABLE_PROGRESSBAR,
            "impute_function": impute,
            "profiling_sorting": PROFILING_SORTING,
            "profiling_filename": PROFILING_FILENAME,
            "profiling": PROFILING
        }
        for attr, value in defaults.items():
            if getattr(self, attr) is None:
                setattr(self, attr, value)

        allowed_default_fc_parameter_strings = {
            "minimal": MinimalFCParameters(),
            "efficient": EfficientFCParameters(),
            "comprehensive": ComprehensiveFCParameters()
        }
        if isinstance(self.default_fc_parameters, str):
            if self.default_fc_parameters not in allowed_default_fc_parameter_strings:
                raise ValueError(f"If `default_fc_parameters` is passed as a string, it must be one of"
                                 f" {allowed_default_fc_parameter_strings}, but found: {self.default_fc_parameters}")
            else:
                self.default_fc_parameters = allowed_default_fc_parameter_strings[self.default_fc_parameters]


class TSFreshFeatureExtractor(BaseTSFreshFeatureExtractor):
    """Transformer for extracting features from given timeseries container.
    
    Use:
        >> tf = TsFreshFeatureExtractor()
        >> X_transformed = tf.fit_transform(X_train)

    Dependencies: 
        tsfresh
    """

    def transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame
        """
        # input checks
        validate_X(X)
        Xt = from_nested_to_long(X)
        check_is_fitted(self, "_is_fitted")

        from tsfresh import extract_features
        Xt = extract_features(Xt, column_id="index", column_value="value", column_kind="column",
                              column_sort="time_index", default_fc_parameters=self.default_fc_parameters,
                              kind_to_fc_parameters=self.kind_to_fc_parameters, n_jobs=self.n_jobs,
                              show_warnings=self.show_warnings, disable_progressbar=self.disable_progressbar,
                              impute_function=self.impute_function, profile=self.profiling,
                              profiling_filename=self.profiling_filename, profiling_sorting=self.profiling_sorting,
                              distributor=self.distributor)

        if self.keep_original:
            return pd.merge(X, Xt, left_index=True, right_index=True, how="left")
        else:
            return Xt


class TSFreshRelevantFeatureExtractor(BaseTSFreshFeatureExtractor):

    """Transformer for extracting features from given timeseries container.
        Use:
        >> tf = TsFreshRelevantFeatureExtractor()
        >> X_transformed = tf.fit_transform(X_train, y_train)

        Dependencies: 
        tsfresh
    """

    def __init__(self, default_fc_parameters="comprehensive", kind_to_fc_parameters=None, chunksize=None,
                 n_jobs=None, show_warnings=None, disable_progressbar=None,
                 impute_function=None, profiling=None, profiling_filename=None,
                 profiling_sorting=None, distributor=None, keep_time_series=False,
                 test_for_binary_target_binary_feature=None, test_for_binary_target_real_feature=None,
                 test_for_real_target_binary_feature=None, test_for_real_target_real_feature=None, fdr_level=None,
                 hypotheses_independent=None,
                 ml_task='auto'):

        super(TSFreshRelevantFeatureExtractor, self).__init__(
            default_fc_parameters=default_fc_parameters,
            kind_to_fc_parameters=kind_to_fc_parameters,
            chunksize=chunksize,
            n_jobs=n_jobs,
            show_warnings=show_warnings,
            disable_progressbar=disable_progressbar,
            impute_function=impute_function,
            profiling=profiling,
            profiling_filename=profiling_filename,
            profiling_sorting=profiling_sorting,
            distributor=distributor,
            keep_time_series=keep_time_series,
        )
        self.test_for_binary_target_binary_feature = test_for_binary_target_binary_feature
        self.test_for_binary_target_real_feature = test_for_binary_target_real_feature
        self.test_for_real_target_binary_feature = test_for_real_target_binary_feature
        self.test_for_real_target_real_feature = test_for_real_target_real_feature
        self.fdr_level = fdr_level
        self.hypotheses_independent = hypotheses_independent
        self.ml_task = ml_task

    def _set_selection_defaults(self):
        """Helper function to set default values from tsfresh.

        Dependencies: 
            tsfresh.defaults
        
        """
        from tsfresh.defaults import TEST_FOR_BINARY_TARGET_BINARY_FEATURE
        from tsfresh.defaults import TEST_FOR_BINARY_TARGET_REAL_FEATURE
        from tsfresh.defaults import TEST_FOR_REAL_TARGET_BINARY_FEATURE
        from tsfresh.defaults import TEST_FOR_REAL_TARGET_REAL_FEATURE
        from tsfresh.defaults import FDR_LEVEL

        defaults = {
            "test_for_binary_target_binary_feature": TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
            "test_for_binary_target_real_feature": TEST_FOR_BINARY_TARGET_REAL_FEATURE,
            "test_for_real_target_binary_feature": TEST_FOR_REAL_TARGET_BINARY_FEATURE,
            "test_for_real_target_real_feature": TEST_FOR_REAL_TARGET_REAL_FEATURE,
            "fdr_level": FDR_LEVEL
        }
        for attr, value in defaults.items():
            if getattr(self, attr) is None:
                setattr(self, attr, value)

    def fit(self, X, y=None):
        """Fit.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        y : pd.Series or np.array
            Target variable

        Returns
        -------
        self : an instance of self
        """

        from tsfresh.transformers.feature_selector import FeatureSelector

        # input checks
        validate_X(X)
        if y is None:
            raise ValueError(f"{self.__class__.__name__} requires `y` in `fit`.")
        validate_y(y)
        self._set_extraction_defaults()
        self._set_selection_defaults()

        self.feature_extractor_ = TSFreshFeatureExtractor(
            default_fc_parameters=self.default_fc_parameters,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            chunksize=self.chunksize,
            n_jobs=self.n_jobs,
            show_warnings=self.show_warnings,
            disable_progressbar=self.disable_progressbar,
            profiling=self.profiling,
            profiling_filename=self.profiling_filename,
            profiling_sorting=self.profiling_sorting,
            keep_time_series=False,
        )
        self.feature_selector_ = FeatureSelector(
            test_for_binary_target_binary_feature=self.test_for_binary_target_binary_feature,
            test_for_binary_target_real_feature=self.test_for_binary_target_real_feature,
            test_for_real_target_binary_feature=self.test_for_real_target_binary_feature,
            test_for_real_target_real_feature=self.test_for_real_target_real_feature,
            fdr_level=self.fdr_level,
            hypotheses_independent=self.hypotheses_independent,
            n_jobs=self.n_jobs,
            chunksize=self.chunksize,
            ml_task=self.ml_task
        )
        Xt = self.feature_extractor_.fit_transform(X)
        self.feature_selector_.fit(Xt, y)
        return self

    def transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        y : pd.Series or np.array
            Target variable

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame
        """
        check_is_fitted(self, ["feature_selector_", "feature_extractor_"])
        validate_X(X)
        Xt = self.feature_extractor_.transform(X)
        Xt = self.feature_selector_.transform(Xt)
        if self.keep_original:
            return pd.merge(X, Xt, left_index=True, right_index=True, how="left")
        else:
            return Xt
