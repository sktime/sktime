import numpy as np
import pandas as pd
from functools import partial
from sklearn.utils.validation import check_is_fitted


from sktime.transformers.base import BaseTransformer
from sktime.transformers.segment import RandomIntervalSegmenter
from sktime.utils.data_container import check_equal_index, convert_data
from sktime.utils.validation.supervised import validate_X, check_X_is_univariate

import tsfresh
from tsfresh import extract_features, extract_relevant_features, select_features,defaults
from tsfresh.feature_extraction import ComprehensiveFCParameters,MinimalFCParameters,EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.transformers.feature_selector import FeatureSelector
from tsfresh.utilities.dataframe_functions import impute_dataframe_range, get_range_values_per_column

COMPREHENSIVE = "COMPREHENSIVE"
MINIMAL = "MINIMAL"
EFFICIENT = "EFFICIENT"

class PlateauFinder(BaseTransformer):
    """Transformer that finds segments of the same given value, plateau in the time series, and
    returns the starting indices and lengths.

    Parameters
    ----------
    value : {int, float, np.nan, np.inf}
        Value for which to find segments
    min_length : int
        Minimum lengths of segments with same value to include.
        If min_length is set to 1, the transformer can be used as a value finder.
    """

    def __init__(self, value=np.nan, min_length=2):
        self.value = value
        self.min_length = min_length

        self._starts = []
        self._lengths = []

    def transform(self, X, y=None):
        """Transform X.
        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_columns]
            Nested dataframe with time-series in cells.
        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame
        """

        # input checks
        validate_X(X)
        check_X_is_univariate(X)

        # get column name
        column_name = X.columns[0]

        # find plateaus (segments of the same value)
        for x in X.iloc[:, 0]:
            x = np.asarray(x)

            # find indices of transition
            if np.isnan(self.value):
                i = np.where(np.isnan(x), 1, 0)

            elif np.isinf(self.value):
                i = np.where(np.isinf(x), 1, 0)

            else:
                i = np.where(x == self.value, 1, 0)

            # pad and find where segments transition
            transitions = np.diff(np.hstack([0, i, 0]))

            # compute starts, ends and lengths of the segments
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            lengths = ends - starts

            # filter out single points
            starts = starts[lengths >= self.min_length]
            lengths = lengths[lengths >= self.min_length]

            self._starts.append(starts)
            self._lengths.append(lengths)

        # put into dataframe
        Xt = pd.DataFrame()
        column_prefix = "%s_%s" % (column_name, "nan" if np.isnan(self.value) else str(self.value))
        Xt["%s_starts" % column_prefix] = pd.Series(self._starts)
        Xt["%s_lengths" % column_prefix] = pd.Series(self._lengths)
        return Xt


class DerivativeSlopeTransformer(BaseTransformer):
    # TODO add docstrings
    def transform(self, X, y=None):
        num_cases, num_dim = X.shape
        output_df = pd.DataFrame()
        for dim in range(num_dim):
            dim_data = X.iloc[:, dim]
            out = DerivativeSlopeTransformer.row_wise_get_der(dim_data)
            output_df['der_dim_' + str(dim)] = pd.Series(out)

        return output_df

    @staticmethod
    def row_wise_get_der(X):

        def get_der(x):
            der = []
            for i in range(1, len(x) - 1):
                der.append(((x[i] - x[i - 1]) + ((x[i + 1] - x[i - 1]) / 2)) / 2)
            return pd.Series([der[0]] + der + [der[-1]])

        return [get_der(x) for x in X]


class RandomIntervalFeatureExtractor(RandomIntervalSegmenter):
    """
    Transformer that segments time-series into random intervals
    and subsequently extracts series-to-primitives features from each interval.

    n_intervals: str{'sqrt', 'log', 'random'}, int or float, optional (default='sqrt')
        Number of random intervals to generate, where m is length of time series:
        - If "log", log of m is used.
        - If "sqrt", sqrt of m is used.
        - If "random", random number of intervals is generated.
        - If int, n_intervals intervals are generated.
        - If float, int(n_intervals * m) is used with n_intervals giving the fraction of intervals of the
        time series length.

        For all arguments relative to the length of the time series, the generated number of intervals is
        always at least 1.

    features: list of functions, optional (default=None)
        Applies each function to random intervals to extract features.
        If None, only the mean is extracted.

    random_state: : int, RandomState instance, optional (default=None)
        - If int, random_state is the seed used by the random number generator;
        - If RandomState instance, random_state is the random number generator;
        - If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, n_intervals='sqrt', min_length=2, features=None, random_state=None):
        super(RandomIntervalFeatureExtractor, self).__init__(
            n_intervals=n_intervals,
            min_length=min_length,
            random_state=random_state,
        )

        # Check input of feature calculators, i.e list of functions to be applied to time-series
        if features is None:
            self.features = [np.mean]
        elif isinstance(features, list) and all([callable(func) for func in features]):
            self.features = features
        else:
            raise ValueError('Features must be list containing only functions (callables) to be '
                             'applied to the data columns')

    def transform(self, X, y=None):
        """
        Transform X, segments time-series in each column into random intervals using interval indices generated
        during `fit` and extracts features from each interval.

        Parameters
        ----------
        X : nested pandas.DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas.DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """
        # Check is fit had been called
        check_is_fitted(self, 'intervals_')
        validate_X(X)
        check_X_is_univariate(X)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns of input is different from what was seen'
                             'in `fit`')
        # Input validation
        # if not all([np.array_equal(fit_idx, trans_idx) for trans_idx, fit_idx in zip(check_equal_index(X),
        #                                                                              self._time_index)]):
        #     raise ValueError('Indexes of input time-series are different from what was seen in `fit`')

        n_rows, n_columns = X.shape
        n_features = len(self.features)

        n_intervals = len(self.intervals_)

        # Compute features on intervals.
        Xt = np.zeros((n_rows, n_features * n_intervals))  # Allocate output array for transformed data
        self.columns_ = []
        colname = X.columns[0]

        # Tabularize each column assuming series have equal indexes in any given column.
        # TODO generalise to non-equal-index cases
        arr = tabularize(X, return_array=True)
        i = 0
        for func in self.features:
            # TODO generalise to series-to-series functions and function kwargs
            for start, end in self.intervals_:
                interval = arr[:, start:end]

                # Try to use optimised computations over axis if possible, otherwise iterate over rows.
                try:
                    Xt[:, i] = func(interval, axis=1)
                except TypeError as e:
                    if str(e) == f"{func.__name__}() got an unexpected keyword argument 'axis'":
                        Xt[:, i] = np.apply_along_axis(func, 1, interval)
                    else:
                        raise
                i += 1
                self.columns_.append(f'{colname}_{start}_{end}_{func.__name__}')

        Xt = pd.DataFrame(Xt)
        Xt.columns = self.columns_
        return Xt


class TsFreshFeatureExtractor(BaseTransformer):
    
    def __init__(self,default_fc_parameters=None,kind_to_fc_parameters=None,
                    chunksize=defaults.CHUNKSIZE,
                     n_jobs=defaults.N_PROCESSES, show_warnings=defaults.SHOW_WARNINGS,
                     disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                     impute_function=defaults.IMPUTE_FUNCTION,
                     profile=defaults.PROFILING,
                     profiling_filename=defaults.PROFILING_FILENAME,
                     profiling_sorting=defaults.PROFILING_SORTING,
                     distributor=None):
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.impute_function = impute_function
        self.profile = profile
        self.profiling_sorting = profiling_sorting
        self.profiling_filename = profiling_filename
        self.distributor = distributor
        self.passed_default_fc_params = None
        self.passed_kind_to_fc_params = kind_to_fc_parameters

        #time series attribute
        self.timeseries_container = None

    def set_timeseries_container(self, timeseries_container):
        self.timeseries_container = timeseries_container    

    def _check_default_rc_parameters(self):

        if self.default_fc_parameters is None:
            return

        if not (isinstance(self.default_fc_parameters,str) or isinstance(self.default_fc_parameters,dict) or isinstance(self.default_fc_parameters,
                (tsfresh.feature_extraction.settings.ComprehensiveFCParameters,
                tsfresh.feature_extraction.settings.MinimalFCParameters,
                tsfresh.feature_extraction.settings.EfficientFCParameters))):
            raise ValueError("default_fc_parameters must be either of the predefined classes or of type dict or a string")
        
        #TODO remove valuerror in future versions
        if isinstance(self.default_fc_parameters,str):
            if self.default_fc_parameters == COMPREHENSIVE:
                self.passed_default_fc_params = tsfresh.feature_extraction.settings.ComprehensiveFCParameters()
            elif self.default_fc_parameters == MINIMAL:
                self.passed_default_fc_params = tsfresh.feature_extraction.settings.MinimalFCParameters()
            elif self.default_fc_parameters == EFFICIENT:
                self.passed_default_fc_params = tsfresh.feature_extraction.settings.EfficientFCParameters()
            else:
                raise ValueError("Unsupported default_fc_parameters, choose one of COMPREHENSIVE,MINIMAL or EFFICIENT")


        elif isinstance(self.default_fc_parameters,
            (tsfresh.feature_extraction.settings.ComprehensiveFCParameters,
            tsfresh.feature_extraction.settings.MinimalFCParameters,
            tsfresh.feature_extraction.settings.EfficientFCParameters)):
            self.passed_default_fc_params = self.default_fc_parameters

            # TODO checks to be performed over custom parameters
        elif isinstance(self.default_fc_parameters,dict):
            if len(self.default_fc_parameters) == 0:
                self.passed_default_fc_params = None
            else:
                self.passed_default_fc_params = self.default_fc_parameters

        else:
            raise ValueError("Invalid type of default_fc_parameters")

    def fit(self,X, y=None):
        #empty
        return self

    def transform(self, X, y=None):
        """Transform X.
        Parameters
        ----------
        X : univariate nested pandas DataFrame of shape [n_samples, n_columns]
            Nested dataframe with time-series in cells.
        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame
        """
        #input checks
        validate_X(X)
        # check_X_is_univariate(X)
        X_time_series = convert_data(X)
       

        # check for default_fc_parameters
        # TODO Is this method required? Drop handling this error to tsfresh?
        self._check_default_rc_parameters()
        
        # TODO Checks for kind_to_fc_params

        Xt = extract_features(
                    X_time_series,
                    column_id="index", column_value="value", 
                    column_kind="column", column_sort="time_index",
                    default_fc_parameters=self.passed_default_fc_params,
                    kind_to_fc_parameters=self.passed_kind_to_fc_params,
                    n_jobs=self.n_jobs, show_warnings=self.show_warnings,
                    disable_progressbar=self.disable_progressbar,
                    impute_function=self.impute_function,
                    profile=self.profile,
                    profiling_filename=self.profiling_filename,
                    profiling_sorting=self.profiling_sorting,
                    distributor=self.distributor)

        return Xt


    def transform_with_input(self,X,y=None):
        #input checks
        validate_X(X)

        # set timeseries_container
        if self.timeseries_container is None:
            raise RuntimeError("You have to provide a time series using the set_timeseries_container function before.")

        # check_X_is_univariate(X)
        timeseries_container_converted = convert_data(self.timeseries_container)
        
        self._check_default_rc_parameters()
        extracted_features = extract_features(timeseries_container_converted,
                                                column_id="index", column_value="value", 
                                                column_kind="column", column_sort="time_index",
                                                default_fc_parameters=self.passed_default_fc_params,
                                                kind_to_fc_parameters=self.passed_kind_to_fc_params,
                                                n_jobs=self.n_jobs, show_warnings=self.show_warnings,
                                                disable_progressbar=self.disable_progressbar,
                                                impute_function=self.impute_function,
                                                profile=self.profile,
                                                profiling_filename=self.profiling_filename,
                                                profiling_sorting=self.profiling_sorting,
                                                distributor=self.distributor)

        X_merged = pd.merge(X, extracted_features, left_index=True, right_index=True, how="left")

        return X_merged



class TsFreshFeatureSelector(BaseTransformer):
    def __init__(self,
                 filter_only_tsfresh_features=True,
                 default_fc_parameters=None,
                 kind_to_fc_parameters=None,
                 chunksize=defaults.CHUNKSIZE,
                 n_jobs=defaults.N_PROCESSES,
                 show_warnings=defaults.SHOW_WARNINGS,
                 disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                 profile=defaults.PROFILING,
                 profiling_filename=defaults.PROFILING_FILENAME,
                 profiling_sorting=defaults.PROFILING_SORTING,
                 test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                 test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                 test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                 test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                 fdr_level=defaults.FDR_LEVEL,
                 hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
                 ml_task='auto'):
        self.filter_only_tsfresh_features = filter_only_tsfresh_features
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        # self.column_id = column_id
        # self.column_sort = column_sort
        # self.column_kind = column_kind
        # self.column_value = column_value
        # self.timeseries_container = timeseries_container
        self.chunksize = chunksize
        self.n_jobs = n_jobs
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.profile = profile
        self.profiling_filename = profiling_filename
        self.profiling_sorting = profiling_sorting
        self.test_for_binary_target_binary_feature = test_for_binary_target_binary_feature
        self.test_for_binary_target_real_feature = test_for_binary_target_real_feature
        self.test_for_real_target_binary_feature = test_for_real_target_binary_feature
        self.test_for_real_target_real_feature = test_for_real_target_real_feature
        self.fdr_level = fdr_level
        self.hypotheses_independent = hypotheses_independent
        self.ml_task = ml_task


        # extractor and selector
        self.feature_extractor = None
        self.feature_selector = None
    
    def fit(self,X,y):

        #input checks
        validate_X(X)

        # if self.timeseries_container is None:
        #     raise RuntimeError("You have to provide a time series using the set_timeseries_container function before.")

        self.feature_extractor = TsFreshFeatureExtractor(
            default_fc_parameters=self.default_fc_parameters,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            chunksize=self.chunksize,
            n_jobs=self.n_jobs,
            show_warnings=self.show_warnings,
            disable_progressbar=self.disable_progressbar,
            profile=self.profile,
            profiling_filename=self.profiling_filename,
            profiling_sorting=self.profiling_sorting)


        self.feature_selector = FeatureSelector(
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

        # Check to include old features
        X_old = X

        #Transform timseries data
        X_transformed = self.feature_extractor.transform(X_old)
        print(X_transformed.head())

        self.col_to_max, self.col_to_min, self.col_to_median = get_range_values_per_column(X_transformed)

        #Immpute timeseries data
        X_transformed = impute_dataframe_range(X_transformed, col_to_max=self.col_to_max, col_to_median=self.col_to_median,
                                             col_to_min=self.col_to_min)

        # print("Imputed X is",X_transformed.head())
        # Fit using transformed timeseries data and y
        self.feature_selector.fit(X_transformed, y)

        return self


    def transform(self,X):
        if self.feature_selector is None:
            raise RuntimeError("You have to call fit before calling transform.")

        if self.feature_selector.relevant_features is None:
            raise RuntimeError("You have to call fit before calling transform.")

        relevant_time_series_features = set(self.feature_selector.relevant_features) - set(pd.DataFrame(X).columns)
        relevant_extraction_settings = from_columns(relevant_time_series_features)

        # Set imputing strategy
        #TODO keeo the impute strategy defaults.IMPUTE_FUNCTION or set it customised?
        impute_function = partial(impute_dataframe_range, col_to_max=self.col_to_max,
                                  col_to_min=self.col_to_min, col_to_median=self.col_to_median)
        relevant_feature_extractor = TsFreshFeatureExtractor(kind_to_fc_parameters=relevant_extraction_settings,
                                                      chunksize=self.feature_extractor.chunksize,
                                                      n_jobs=self.feature_extractor.n_jobs,
                                                      show_warnings=self.feature_extractor.show_warnings,
                                                      disable_progressbar=self.feature_extractor.disable_progressbar,
                                                      impute_function=impute_function,
                                                      profile=self.feature_extractor.profile,
                                                      profiling_filename=self.feature_extractor.profiling_filename,
                                                      profiling_sorting=self.feature_extractor.profiling_sorting)
        
        Xt = relevant_feature_extractor.transform(X)

        return Xt 


    def fit_transform(self, X, y):
        if y is None:
            raise RuntimeError("You have to pass in y")
        
        #input checks
        validate_X(X)

        return self.fit(X, y).transform(X)