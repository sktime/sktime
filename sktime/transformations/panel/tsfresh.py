# -*- coding: utf-8 -*-
"""tsfresh interface class."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["AyushmaanSeth", "Markus Löning", "Alwin Wang"]
__all__ = ["TSFreshFeatureExtractor", "TSFreshRelevantFeatureExtractor"]

from warnings import warn

from sktime.datatypes._panel._convert import from_nested_to_long
from sktime.transformations.base import BaseTransformer
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("tsfresh", severity="warning")


class _TSFreshFeatureExtractor(BaseTransformer):
    """Base adapter class for tsfresh transformations."""

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "python_dependencies": "tsfresh",
        "python_version": "<3.10",
    }

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

        super(_TSFreshFeatureExtractor, self).__init__()

        # _get_extraction_params should be after the init because this imports tsfresh
        # and the init checks for python version and tsfresh being present
        self.default_fc_parameters_ = None
        self.default_fc_parameters_ = self._get_extraction_params()

    def _get_extraction_params(self):
        """Set default parameters from tsfresh."""
        # make n_jobs compatible with scikit-learn
        self.n_jobs = check_n_jobs(self.n_jobs)

        # lazy imports to avoid hard dependency
        from tsfresh.defaults import (
            CHUNKSIZE,
            DISABLE_PROGRESSBAR,
            N_PROCESSES,
            PROFILING,
            PROFILING_FILENAME,
            PROFILING_SORTING,
            SHOW_WARNINGS,
        )
        from tsfresh.feature_extraction.settings import (
            ComprehensiveFCParameters,
            EfficientFCParameters,
            MinimalFCParameters,
        )
        from tsfresh.utilities.dataframe_functions import impute

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
    """Transformer for extracting time series features.

    References
    ----------
    .. [1]  https://github.com/blue-yonder/tsfresh
    """

    _tags = {
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
    }

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series
            transformed version of X
        """
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
        return {"disable_progressbar": True, "show_warnings": False}


class TSFreshRelevantFeatureExtractor(_TSFreshFeatureExtractor):
    """Transformer for extracting and selecting features.

    References
    ----------
    .. [1]  https://github.com/blue-yonder/tsfresh
    """

    _tags = {
        "requires_y": True,  # does y need to be passed in fit?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd_Series_Table",
        # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
    }

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
        """Set default values from tsfresh."""
        # lazy imports to avoid hard dependency
        from tsfresh.defaults import (
            FDR_LEVEL,
            HYPOTHESES_INDEPENDENT,
            TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
            TEST_FOR_BINARY_TARGET_REAL_FEATURE,
            TEST_FOR_REAL_TARGET_BINARY_FEATURE,
            TEST_FOR_REAL_TARGET_REAL_FEATURE,
        )

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

    def _fit(self, X, y=None):
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
        return self

    def _transform(self, X, y=None):
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
        Xt = self.extractor_.transform(X)
        Xt = self.selector_.transform(Xt)
        return Xt.reindex(X.index)

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
        params = {
            "disable_progressbar": True,
            "show_warnings": False,
            "fdr_level": 0.01,
        }
        return params
