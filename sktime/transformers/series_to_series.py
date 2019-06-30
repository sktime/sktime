from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_random_state
import numpy as np
import pandas as pd
from sktime.utils.validation import check_equal_index
from sktime.utils.transformations import tabularize, detabularize, concat_nested_arrays, remove_trend, add_trend
from sktime.transformers.base import BaseTransformer
from sktime.utils.seasonality import remove_seasonality, add_seasonality


__all__ = ['RandomIntervalSegmenter',
           'IntervalSegmenter',
           'DerivativeSlopeTransformer',
           'TimeSeriesConcatenator',
           'Detrender',
           "Deseasonaliser",
           'Deseasonalizer']

__author__ = ["Markus Löning", "Jason Lines"]


class IntervalSegmenter(BaseTransformer):
    """
    Interval segmentation transformer.

    Parameters
    ----------
    intervals : int, np.ndarray or list of np.ndarrays with one for each column of input data.
        Intervals to generate.
        - If int, intervals are generated.
        - If ndarray, 2d np.ndarray [n_intervals, 2] with rows giving intervals, the first column giving start points,
        and the second column giving end points of intervals
        - If list of np.ndarrays, there is one array for each column.
    """

    def __init__(self, intervals=None, check_input=True):
        self.intervals = intervals
        self.check_input = check_input
        self.input_indexes_ = []  # list of time-series indexes of each column
        self.input_shape_ = ()

    def fit(self, X, y=None):
        """
        Fit transformer, generating random interval indices.

        Parameters
        ----------
        X : pandas DataFrame of shape [n_samples, n_features]
            Input data
        y : pandas Series, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : an instance of self.
        """

        if self.check_input:
            pass
            # TODO check input is series column, not column of primitives

        self.input_shape_ = X.shape

        # Retrieve time-series indexes from each column.
        # TODO generalise to columns with series of unequal length
        self.input_indexes_ = [X.iloc[0, c].index if hasattr(X.iloc[0, c], 'index')
                               else np.arange(X.iloc[0, c].shape[0]) for c in range(self.input_shape_[1])]

        if isinstance(self.intervals, np.ndarray):
            self.intervals_ = [self.intervals]

        if (isinstance(self.intervals, list) and (len(self.intervals) == self.input_shape_[1])
                and np.all([isinstance(intervals, np.ndarray) for intervals in self.intervals])):
            self.intervals_ = self.intervals

        elif np.issubdtype(self.intervals, np.integer):
            self.intervals_ = [np.array_split(self.input_indexes_[c], self.intervals)
                               for c in range(self.input_shape_[1])]

        else:
            raise ValueError(f"`intervals` must be either an integer, a single array or list of arrays with "
                             f"start and end points, but found: {self.intervals}")

        return self

    def transform(self, X, y=None):
        """
        Transform X, segments time-series in each column into random intervals using interval indices generated
        during `fit`.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """

        # Check is fit had been called
        check_is_fitted(self, 'intervals_')

        # Check inputs.
        if self.check_input:
            # Check that the input is of the same shape as the one passed
            # during fit.
            if (X.shape[1] if X.ndim == 2 else 1) != self.input_shape_[1]:
                raise ValueError('Number of columns of input is different from what was seen'
                                 'in `fit`')
            # Input validation
            if not all([np.array_equal(fit_idx, trans_idx) for trans_idx, fit_idx in zip(check_equal_index(X),
                                                                                         self.input_indexes_)]):
                raise ValueError('Indexes of input time-series are different from what was seen in `fit`')

        # Segment into intervals.
        intervals = []
        self.colnames_ = []
        for c, (colname, col) in enumerate(X.items()):

            # Tabularize each column assuming series have equal indexes in any given column.
            # TODO generalise to non-equal-index cases
            arr = tabularize(col, return_array=True)
            for start, end in self.intervals_[c]:
                interval = arr[:, start:end]
                intervals.append(interval)
                self.colnames_.append(f'{colname}_{start}_{end}')

        # Return nested pandas DataFrame.
        Xt = pd.DataFrame(concat_nested_arrays(intervals, return_arrays=True))
        Xt.columns = self.colnames_
        return Xt


class RandomIntervalSegmenter(IntervalSegmenter):
    """Transformer that segments time-series into random intervals with random starting points and lengths. Some
    intervals may overlap and may be duplicates.

    Parameters
    ----------
    n_intervals : str, int or float
        Number of intervals to generate.
        - If "log", log of m is used where m is length of time series.
        - If "sqrt", sqrt of m is used.
        - If "random", random number of intervals is generated.
        - If int, n_intervals intervals are generated.
        - If float, int(n_intervals * m) is used with n_intervals giving the fraction of intervals of the
        time series length.

        For all arguments relative to the length of the time series, the generated number of intervals is
        always at least 1.

        Default is "sqrt".

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    check_input : bool, optional (default=True)
        When set to ``True``, inputs will be validated, otherwise inputs are assumed to be valid
        and no checks are performed. Use with caution.
    """

    def __init__(self, n_intervals='sqrt', min_length=None, random_state=None, check_input=True):

        self.min_length = 1 if min_length is None else min_length
        self.n_intervals = n_intervals
        self.random_state = random_state

        super(RandomIntervalSegmenter, self).__init__(check_input=check_input)

    @property
    def random_state(self):
        """
        Makes private attribute read-only.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        """
        Set random state making sure rng is always updated as well.
        """
        self._random_state = random_state
        self._rng = check_random_state(random_state)

    def fit(self, X, y=None):
        """
        Fit transformer, generating random interval indices.

        Parameters
        ----------
        X : pandas DataFrame of shape [n_samples, n_features]
            Input data
        y : pandas Series, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : RandomIntervalSegmenter
            This estimator
        """

        if self.check_input:
            # TODO check input is series column, not column of primitives
            pass

        self.input_shape_ = X.shape

        # Retrieve time-series indexes from each column.
        # TODO generalise to columns with series of unequal length
        self.input_indexes_ = [X.iloc[0, c].index if hasattr(X.iloc[0, c], 'index')
                               else np.arange(X.iloc[0, c].shape[0]) for c in range(self.input_shape_[1])]

        # Compute random intervals for each column.
        # TODO if multiple columns are passed, introduce option to compute one set of shared intervals,
        #  or rely on ColumnTransformer?
        if self.n_intervals == 'random':
            self.intervals_ = [self._rand_intervals_rand_n(self.input_indexes_[c])
                               for c in range(self.input_shape_[1])]
        else:
            self.intervals_ = [self._rand_intervals_fixed_n(self.input_indexes_[c], n=self.n_intervals)
                               for c in range(self.input_shape_[1])]
        return self

    def _rand_intervals_rand_n(self, x):
        """
        Compute a random number of intervals from index (x) with
        random starting points and lengths. Intervals are unique, but may overlap.

        Parameters
        ----------
        x : array_like, shape = [n_observations]

        Returns
        -------
        intervals : array-like, shape = [n, 2]
            2d array containing start and end points of intervals

        References
        ----------
        ..[1] Deng, Houtao, et al. "A time series forest for classification and feature extraction."
            Information Sciences 239 (2013): 142-153.
        """

        starts = []
        ends = []
        m = x.shape[0]  # series length
        W = self._rng.randint(1, m, size=int(np.sqrt(m)))
        for w in W:
            size = m - w + 1
            start = self._rng.randint(size, size=int(np.sqrt(size)))
            starts.extend(start)
            for s in start:
                end = s + w
                ends.append(end)
        return np.column_stack([starts, ends])

    def _rand_intervals_fixed_n(self, x, n):
        """
        Compute a fixed number (n) of intervals from index (x) with
        random starting points and lengths. Intervals may overlap and may not be unique.

        Parameters
        ----------
        x : array_like, shape = [n_observations]
            Array containing the time-series index.
        n : 'sqrt', 'log', float or int

        Returns
        -------
        intervals : array-like, shape = [n, 2]
            2d array containing start and end points of intervals
        """

        m = len(x)
        # compute number of random intervals relative to series length (m)
        # TODO use smarter dispatch at construction to avoid evaluating if-statements here each time function is called
        if np.issubdtype(type(n), np.integer) and (n >= 1):
            pass
        elif n == 'sqrt':
            n = int(np.sqrt(m))
        elif n == 'log':
            n = int(np.log(m))
        elif np.issubdtype(type(n), np.floating) and (n > 0) and (n <= 1):
            n = int(m * n)
        else:
            raise ValueError(f'Number of intervals must be either "random", "sqrt", a positive integer, or a float '
                             f'value between 0 and 1, but found {n}.')

        # make sure there is at least one interval
        n = np.maximum(1, n)

        starts = self._rng.randint(m - self.min_length + 1, size=n)
        if n == 1:
            starts = [starts]  # make it an iterable

        ends = [start + self._rng.randint(self.min_length, m - start + 1) for start in starts]
        return np.column_stack([starts, ends])

      
class DerivativeSlopeTransformer(BaseTransformer):
    # TODO add docstrings
    def transform(self, X, y=None):
        num_cases, num_dim = X.shape
        output_df = pd.DataFrame()
        for dim in range(num_dim):
            dim_data = X.iloc[:,dim]
            out = DerivativeSlopeTransformer.row_wise_get_der(dim_data)
            output_df['der_dim_'+str(dim)] = pd.Series(out)

        return output_df

    @staticmethod
    def row_wise_get_der(X):

        def get_der(x):
            der = []
            for i in range(1, len(x) - 1):
                der.append(((x[i] - x[i - 1]) + ((x[i + 1] - x[i - 1]) / 2)) / 2)
            return pd.Series([der[0]] + der + [der[-1]])

        return [get_der(x) for x in X]


class TimeSeriesConcatenator(BaseTransformer):
    """Transformer that concatenates multivariate time series/panel data into long univiariate time series/panel
        data by simply concatenating times series in time.
    """

    def transform(self, X, y=None):
        """Concatenate multivariate time series/panel data into long univiariate time series/panel
        data by simply concatenating times series in time.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and single column
        """

        check_is_fitted(self, 'is_fitted_')

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected input is a pandas DataFrame, but found {type(X)}")

        Xt = detabularize(tabularize(X))
        return Xt


class Detrender(BaseTransformer):
    """A transformer that removes trend of given polynomial order from time series/panel data

    Parameters
    ----------
    order : int
        Polynomial order, zero: mean, one: linear, two: quadratic, etc
    check_input : bool, optional (default=True)
        When set to ``True``, inputs will be validated, otherwise inputs are assumed to be valid
        and no checks are performed. Use with caution.
    """

    def __init__(self, order=0, check_input=True):

        if not (isinstance(order, int) and (order >= 0)):
            raise ValueError(f"order must be a positive integer, but found: {type(order)}")
        self.order = order
        self.check_input = check_input
        self.theta = None

    def transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """

        if self.check_input:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"Input must be pandas DataFrame, but found: {type(X)}")

        # TODO work on multiple columns, currently only works on first column
        if X.shape[1] > 1:
            raise NotImplementedError(f"Currently does not work on multiple columns")

        # convert into tabular format
        Xs = tabularize(X.iloc[:, 0])

        # remove trend, keeping fitted polynomial coefficients (theta)
        Xt, self.theta = remove_trend(Xs, order=self.order)

        # convert back into nested format
        Xt = detabularize(pd.DataFrame(Xt))
        Xt.columns = X.columns
        return Xt

    def inverse_transform(self, X, y=None):
        """Inverse transform X

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """

        if self.check_input:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"Input must be pandas DataFrame, but found: {type(X)}")

        # TODO work on multiple columns, currently only works on first column
        if X.shape[1] > 1:
            raise NotImplementedError(f"Currently does not work on multiple columns, make use of ColumnTransformer "
                                      f"instead")

        # convert into tabular format
        Xs = tabularize(X.iloc[:, 0])

        # add trend, keeping fitted polynomial coefficients
        Xit = add_trend(Xs, theta=self.theta)

        # convert back into nested format
        Xit = detabularize(pd.DataFrame(Xit))
        Xit.columns = X.columns
        return Xit


class Deseasonaliser(BaseTransformer):
    """A transformer that removes trend of given polynomial order from time series/panel data

    Parameters
    ----------
    sp : int, optional (default=1)
        Seasonal periodicity
    model : str {'additive', 'multiplicative'}, optional (default='additive')
        Model to use for estimating seasonal component
    check_input : bool, optional (default=True)
        When set to ``True``, inputs will be validated, otherwise inputs are assumed to be valid
        and no checks are performed. Use with caution.
    """

    def __init__(self, sp=1, model='additive', check_input=True):
        self.sp = sp
        self.model = model
        self.check_input = check_input
        self.si = None

    def transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """
        if self.check_input:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"Input must be pandas DataFrame, but found: {type(X)}")

        # TODO work on multiple columns, currently only works on first column
        if X.shape[1] > 1:
            raise NotImplementedError(f"Currently does not work on multiple columns, make use of ColumnTransformer "
                                      f"instead")

        # convert into list of series/rows
        xs = X.iloc[:, 0].tolist()

        n = X.shape[0]  # number of rows
        m = X.iloc[0, 0].shape[0]  # number of time series observations
        xt = np.zeros((n, m))
        self.si = np.zeros((n, self.sp))

        # remove seasonality from each series/row
        # TODO make more efficient/vectorise to work on multiple rows simultaneously
        for i, x in enumerate(xs):
            xt[i, :], self.si[i, :] = remove_seasonality(x, sp=self.sp, model=self.model)

        # convert back into nested format
        Xt = detabularize(pd.DataFrame(xt))
        Xt.columns = X.columns
        return Xt

    def inverse_transform(self, X, y=None):
        """Inverse transform X

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """
        if self.check_input:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"Input must be pandas DataFrame, but found: {type(X)}")

        # TODO work on multiple columns, currently only works on first column
        if X.shape[1] > 1:
            raise NotImplementedError(f"Currently does not work on multiple columns")

        # convert into list of series/rows
        xs = X.iloc[:, 0].tolist()

        n = X.shape[0]  # number of rows
        m = X.iloc[0, 0].shape[0]  # number of time series observations
        xit = np.zeros((n, m))

        # remove seasonality from each series/row
        for i, x in enumerate(xs):
            xit[i, :] = add_seasonality(x, self.si[i, :])

        # convert back into nested format
        Xit = detabularize(pd.DataFrame(xit))
        Xit.columns = X.columns
        return Xit


Deseasonalizer = Deseasonaliser
