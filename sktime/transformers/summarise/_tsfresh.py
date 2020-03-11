#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["", "Markus Löning"]
__all__ = ["TSFreshFeatureExtractor", "TSFreshRelevantFeatureExtractor"]

import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sktime.transformers.base import BaseTransformer
from sktime.utils.data_container import from_nested_to_long
from sktime.utils.validation.supervised import validate_X
from sktime.utils.validation.supervised import validate_y


class BaseTSFreshFeatureExtractor(BaseTransformer):

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
        self.keep_time_series = keep_time_series

    def fit(self, X, y=None):
        validate_X(X)
        self._set_extraction_defaults()
        self._is_fitted = True
        return self

    def _set_extraction_defaults(self):
        """Helper function to set default parameters for feature extraction"""
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

        self.n_jobs = N_PROCESSES if self.n_jobs is None else self.n_jobs
        self.chunksize = CHUNKSIZE if self.chunksize is None else self.chunksize
        self.show_warnings = SHOW_WARNINGS if self.show_warnings is None else self.show_warnings
        self.disable_progressbar = DISABLE_PROGRESSBAR if self.disable_progressbar is None else self.disable_progressbar
        self.impute_function = impute if self.impute_function is None else self.impute_function
        self.profiling_sorting = PROFILING_SORTING if self.profiling_sorting is None else self.profiling_sorting
        self.profiling_filename = PROFILING_FILENAME if self.profiling_filename is None else self.profiling_filename
        self.profiling = PROFILING if self.profiling is None else self.profiling

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

        if self.keep_time_series:
            return pd.merge(X, Xt, left_index=True, right_index=True, how="left")
        else:
            return Xt


class TSFreshRelevantFeatureExtractor(BaseTSFreshFeatureExtractor):

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
        from tsfresh.defaults import TEST_FOR_BINARY_TARGET_BINARY_FEATURE
        from tsfresh.defaults import TEST_FOR_BINARY_TARGET_REAL_FEATURE
        from tsfresh.defaults import TEST_FOR_REAL_TARGET_BINARY_FEATURE
        from tsfresh.defaults import TEST_FOR_REAL_TARGET_REAL_FEATURE
        from tsfresh.defaults import FDR_LEVEL

        self.test_for_binary_target_binary_feature = TEST_FOR_BINARY_TARGET_BINARY_FEATURE if \
            self.test_for_binary_target_binary_feature is None else self.test_for_binary_target_binary_feature
        self.test_for_binary_target_real_feature = TEST_FOR_BINARY_TARGET_REAL_FEATURE if \
            self.test_for_binary_target_real_feature is None else self.test_for_binary_target_real_feature
        self.test_for_real_target_binary_feature = TEST_FOR_REAL_TARGET_BINARY_FEATURE if \
            self.test_for_real_target_binary_feature is None else self.test_for_real_target_binary_feature
        self.test_for_real_target_real_feature = TEST_FOR_REAL_TARGET_REAL_FEATURE if \
            self.test_for_real_target_real_feature is None else self.test_for_real_target_real_feature
        self.fdr_level = FDR_LEVEL if self.fdr_level is None else self.fdr_level

    def fit(self, X, y=None):

        # input checks
        validate_X(X)
        if y is None:
            raise ValueError(f"{self.__class__.__name__} requires `y` in `fit`.")
        validate_y(y)
        self._set_extraction_defaults()
        self._set_selection_defaults()

        feature_extractor = TSFreshFeatureExtractor(
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
        Xt = feature_extractor.fit_transform(X)

        from tsfresh.feature_extraction.settings import from_columns
        self._extracted_features = from_columns(Xt)

        from tsfresh.transformers.feature_selector import FeatureSelector
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
        self.feature_selector_.fit(Xt, y)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "feature_selector_")
        validate_X(X)

        feature_extractor = TSFreshFeatureExtractor(kind_to_fc_parameters=self._extracted_features,
                                                    chunksize=self.chunksize,
                                                    n_jobs=self.n_jobs,
                                                    show_warnings=self.show_warnings,
                                                    disable_progressbar=self.disable_progressbar,
                                                    impute_function=self.impute_function,
                                                    profiling=self.profiling,
                                                    profiling_filename=self.profiling_filename,
                                                    profiling_sorting=self.profiling_sorting)
        Xt = feature_extractor.fit_transform(X)
        Xt = self.feature_selector_.transform(Xt)
        if self.keep_time_series:
            return pd.merge(X, Xt, left_index=True, right_index=True, how="left")
        else:
            return Xt
