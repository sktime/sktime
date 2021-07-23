#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Ayushmaan Seth", "Markus Löning", "Alwin Wang"]
__all__ = ["TSFreshFeatureExtractor", "TSFreshRelevantFeatureExtractor"]

from warnings import warn

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.data_processing import from_nested_to_long
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y

_check_soft_dependencies("tsfresh")


class _TSFreshFeatureExtractor(_PanelToTabularTransformer):
    """Base adapter class for tsfresh transformations"""

    def __init__(
        self,
        default_fc_parameters="efficient",
        kind_to_fc_parameters=None,
        chunksize=None,
        n_jobs=1,
        show_warnings=True,
        disable_progressbar=False,
        impute_function=None,
        profiling=None,
        profiling_filename=None,
        profiling_sorting=None,
        distributor=None,
    ):
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

        self.default_fc_parameters_ = None

        super(_TSFreshFeatureExtractor, self).__init__()

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
        check_X(X, coerce_to_pandas=True)
        self.default_fc_parameters_ = self._get_extraction_params()
        self._is_fitted = True
        return self

    def _get_extraction_params(self):
        """Helper function to set default parameters from tsfresh"""
        # make n_jobs compatible with scikit-learn
        self.n_jobs = check_n_jobs(self.n_jobs)

        # lazy imports to avoid hard dependency
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

        # Set defaults from tsfresh
        extraction_params = {
            "kind_to_fc_parameters": self.kind_to_fc_parameters,
            "n_jobs": N_PROCESSES,
            "chunksize": CHUNKSIZE,
            "show_warnings": SHOW_WARNINGS,
            "disable_progressbar": DISABLE_PROGRESSBAR,
            "impute_function": impute,
            "profiling_sorting": PROFILING_SORTING,
            "profiling_filename": PROFILING_FILENAME,
            "profile": PROFILING,
        }

        # Replace defaults with user defined parameters
        for name in extraction_params.keys():
            if hasattr(self, name):
                value = getattr(self, name)
                if value is not None:
                    extraction_params[name] = value

        # Convert convenience string arguments to tsfresh parameters classes
        fc_param_lookup = {
            "minimal": MinimalFCParameters(),
            "efficient": EfficientFCParameters(),
            "comprehensive": ComprehensiveFCParameters(),
        }
        if isinstance(self.default_fc_parameters, str):
            if self.default_fc_parameters not in fc_param_lookup:
                raise ValueError(
                    f"If `default_fc_parameters` is passed as a "
                    f"string, "
                    f"it must be one of"
                    f" {fc_param_lookup.keys()}, but found: "
                    f"{self.default_fc_parameters}"
                )
            else:
                fc_parameters = fc_param_lookup[self.default_fc_parameters]
        else:
            fc_parameters = self.default_fc_parameters
        extraction_params["default_fc_parameters"] = fc_parameters

        return extraction_params


class TSFreshFeatureExtractor(_TSFreshFeatureExtractor):
    """Transformer for extracting time series features

    References
    ----------
    ..[1]  https://github.com/blue-yonder/tsfresh
    """

    def transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        y : pd.Series, optional (default=None)

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame
        """
        # input checks
        self.check_is_fitted()
        X = check_X(X, coerce_to_pandas=True)

        # tsfresh requires unique index, returns only values for
        # unique index values
        if X.index.nunique() < X.shape[0]:
            warn(
                "tsfresh requires a unique index, but found "
                "non-unique. To avoid this warning, please make sure the index of X "
                "contains only unique values."
            )
            X = X.reset_index(drop=True)

        Xt = from_nested_to_long(X)

        # lazy imports to avoid hard dependency
        from tsfresh import extract_features

        extraction_params = self._get_extraction_params()
        Xt = extract_features(
            Xt,
            column_id="index",
            column_value="value",
            column_kind="column",
            column_sort="time_index",
            **extraction_params,
        )

        # When using the long input format, tsfresh seems to sort the index,
        # here we make sure we return the dataframe in the sort order as the
        # input data
        return Xt.reindex(X.index)


class TSFreshRelevantFeatureExtractor(_TSFreshFeatureExtractor):
    """Transformer for extracting and selecting features.

    References
    ----------
    ..[1]  https://github.com/blue-yonder/tsfresh
    """

    def __init__(
        self,
        default_fc_parameters="efficient",
        kind_to_fc_parameters=None,
        chunksize=None,
        n_jobs=1,
        show_warnings=True,
        disable_progressbar=False,
        impute_function=None,
        profiling=None,
        profiling_filename=None,
        profiling_sorting=None,
        distributor=None,
        test_for_binary_target_binary_feature=None,
        test_for_binary_target_real_feature=None,
        test_for_real_target_binary_feature=None,
        test_for_real_target_real_feature=None,
        fdr_level=None,
        hypotheses_independent=None,
        ml_task="auto",
    ):

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
        )
        self.test_for_binary_target_binary_feature = (
            test_for_binary_target_binary_feature
        )
        self.test_for_binary_target_real_feature = test_for_binary_target_real_feature
        self.test_for_real_target_binary_feature = test_for_real_target_binary_feature
        self.test_for_real_target_real_feature = test_for_real_target_real_feature
        self.fdr_level = fdr_level
        self.hypotheses_independent = hypotheses_independent
        self.ml_task = ml_task

    def _get_selection_params(self):
        """Helper function to set default values from tsfresh"""
        # lazy imports to avoid hard dependency
        from tsfresh.defaults import TEST_FOR_BINARY_TARGET_BINARY_FEATURE
        from tsfresh.defaults import TEST_FOR_BINARY_TARGET_REAL_FEATURE
        from tsfresh.defaults import TEST_FOR_REAL_TARGET_BINARY_FEATURE
        from tsfresh.defaults import TEST_FOR_REAL_TARGET_REAL_FEATURE
        from tsfresh.defaults import FDR_LEVEL
        from tsfresh.defaults import HYPOTHESES_INDEPENDENT

        # Set defaults
        selection_params = {
            "test_for_binary_target_binary_feature": TEST_FOR_BINARY_TARGET_BINARY_FEATURE,  # noqa: E501
            "test_for_binary_target_real_feature": TEST_FOR_BINARY_TARGET_REAL_FEATURE,
            "test_for_real_target_binary_feature": TEST_FOR_REAL_TARGET_BINARY_FEATURE,
            "test_for_real_target_real_feature": TEST_FOR_REAL_TARGET_REAL_FEATURE,
            "fdr_level": FDR_LEVEL,
            "hypotheses_independent": HYPOTHESES_INDEPENDENT,
        }

        # Replace defaults with user defined parameters
        for name in selection_params.keys():
            value = getattr(self, name)
            if value is not None:
                selection_params[name] = value

        return selection_params

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
        # lazy imports to avoid hard dependency
        from tsfresh.transformers.feature_selector import FeatureSelector

        # input checks
        if y is None:
            raise ValueError(f"{self.__class__.__name__} requires `y` in `fit`.")
        X, y = check_X_y(X, y, coerce_to_pandas=True)

        self.extractor_ = TSFreshFeatureExtractor(
            default_fc_parameters=self.default_fc_parameters,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            chunksize=self.chunksize,
            n_jobs=self.n_jobs,
            show_warnings=self.show_warnings,
            disable_progressbar=self.disable_progressbar,
            profiling=self.profiling,
            profiling_filename=self.profiling_filename,
            profiling_sorting=self.profiling_sorting,
        )

        selection_params = self._get_selection_params()
        extraction_param = self._get_extraction_params()
        self.selector_ = FeatureSelector(
            n_jobs=extraction_param["n_jobs"],
            chunksize=extraction_param["chunksize"],
            ml_task=self.ml_task,
            **selection_params,
        )

        Xt = self.extractor_.fit_transform(X)
        self.selector_.fit(Xt, y)
        self._is_fitted = True
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
        self.check_is_fitted()
        X = check_X(X, coerce_to_pandas=True)
        Xt = self.extractor_.transform(X)
        Xt = self.selector_.transform(Xt)
        return Xt.reindex(X.index)
