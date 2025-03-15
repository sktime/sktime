"""Supervised interval features.

A transformer for the extraction of features on intervals extracted from a supervised
process.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["SupervisedIntervals"]

import numpy as np
from sklearn import preprocessing
from sklearn.utils import check_random_state

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation import check_n_jobs


class SupervisedIntervals(BaseTransformer):
    """Supervised interval feature transformer.

    Extracts intervals in fit using the supervised process described in [1].
    Interval sub-series are extracted for each input feature, and the usefulness of that
    feature extracted on an interval is evaluated using the Fisher score metric.
    Intervals are continually split in half, with the better scoring half retained as a
    feature for the transform.

    Multivariate capability is added by running the supervised interval extraction
    process on each dimension of the input data.

    As the extracted interval features are already extracted for the supervised
    evaluation in fit, the fit_transform method is recommended if the transformed fit
    data is required.

    Parameters
    ----------
    n_intervals : int, default=50
        The number of times the supervised interval selection process is run.
        Each supervised extraction will output a varying amount of features based on
        series length, number of dimensions and the number of features.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    features : function with a single 2d array-like parameter or list of said functions,
            default=None
        Functions used to extract features from selected intervals. If None, defaults to
        the following statistics used in [2]:
        [mean, median, std, slope, min, max, iqr, count_mean_crossing,
        count_above_mean].
    randomised_split_point : bool, default=True
        If True, the split point for interval extraction is randomised as is done in [2]
        rather than split in half.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    intervals_ : list of tuples
        Contains information for each feature extracted in fit. Each tuple contains the
        interval start, interval end, interval dimension and the feature extracted.
        Length will be the same as the amount of transformed features.

    See Also
    --------
    RandomIntervals

    Notes
    -----
    Based on the authors (stevcabello) code: https://github.com/stevcabello/r-STSF/

    References
    ----------
    .. [1] Cabello, N., Naghizade, E., Qi, J. and Kulik, L., 2020, November. Fast and
        accurate time series classification through supervised interval search. In 2020
        IEEE International Conference on Data Mining (ICDM) (pp. 948-953). IEEE.
    .. [2] Cabello, N., Naghizade, E., Qi, J. and Kulik, L., 2021. Fast, accurate and
        interpretable time series classification through randomization. arXiv preprint
        arXiv:2105.14876.
    """

    _tags = {
        "authors": ["MatthewMiddlehurst"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "fit_is_empty": False,
        "capability:unequal_length": False,
        "requires_y": True,
        "python_dependencies": ["numba", "joblib"],
    }

    def __init__(
        self,
        n_intervals=50,
        min_interval_length=3,
        features=None,
        randomised_split_point=True,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.features = features
        self.randomised_split_point = randomised_split_point
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.intervals_ = []

        self._min_interval_length = min_interval_length
        self._features = features
        self._transform_features = []
        self._n_jobs = n_jobs

        super().__init__()

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.
        _X : X, coerced copy of X, if remember_data tag is True
            possibly coerced to inner type or update_data compatible type
            by reference, when possible
        model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |   `X`    | `tf-output`  |     type of return     |
            |----------|--------------|------------------------|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        Explicitly, with examples:
            if `X` is `Series` (e.g., `pd.DataFrame`) and `transform-output` is `Series`
                then the return is a single `Series` of the same mtype
                Example: detrending a single series
            if `X` is `Panel` (e.g., `pd-multiindex`) and `transform-output` is `Series`
                then the return is `Panel` with same number of instances as `X`
                    (the transformer is applied to each input Series instance)
                Example: all series in the panel are detrended individually
            if `X` is `Series` or `Panel` and `transform-output` is `Primitives`
                then the return is `pd.DataFrame` with as many rows as instances in `X`
                Example: i-th row of the return has mean and variance of the i-th series
            if `X` is `Series` and `transform-output` is `Panel`
                then the return is a `Panel` object of type `pd-multiindex`
                Example: i-th instance of the output is the i-th window running over `X`
        """
        from joblib import Parallel, delayed

        from sktime.utils.numba.general import z_normalise_series_3d

        self.reset()
        if y is None:
            raise ValueError("SupervisedIntervals requires `y` in `fit`.")
        X, y, metadata = self._check_X_y(X=X, y=y, return_metadata=True)

        y = self._fit_setup(X, y)
        X_norm = z_normalise_series_3d(X)

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X,
                X_norm,
                y,
                i,
                True,
            )
            for i in range(self.n_intervals)
        )

        (
            intervals,
            transformed_intervals,
        ) = zip(*fit)

        self.intervals_ = []
        for i in intervals:
            self.intervals_.extend(i)

        self._transform_features = [True] * len(self.intervals_)

        Xt = transformed_intervals[0]
        for i in range(1, self.n_intervals):
            Xt = np.hstack((Xt, transformed_intervals[i]))

        self._is_fitted = True

        # obtain configs to control input and output control
        configs = self.get_config()
        input_conv = configs["input_conversion"]
        output_conv = configs["output_conversion"]

        if input_conv and output_conv:
            X_out = self._convert_output(Xt, metadata=metadata)
        else:
            X_out = Xt
        return X_out

    def _fit(self, X, y=None):
        from joblib import Parallel, delayed

        from sktime.utils.numba.general import z_normalise_series_3d

        y = self._fit_setup(X, y)
        X_norm = z_normalise_series_3d(X)

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X,
                X_norm,
                y,
                i,
                False,
            )
            for i in range(self.n_intervals)
        )

        (
            intervals,
            _,
        ) = zip(*fit)

        self.intervals_ = []
        for i in intervals:
            self.intervals_.extend(i)

        self._transform_features = [True] * len(self.intervals_)

        return self

    def _transform(self, X, y=None):
        from joblib import Parallel, delayed

        transform = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._transform_intervals)(
                X,
                i,
            )
            for i in range(len(self.intervals_))
        )

        Xt = np.zeros((X.shape[0], len(transform)))
        for i, t in enumerate(transform):
            Xt[:, i] = t

        return Xt

    def _fit_setup(self, X, y):
        from sktime.utils.numba.stats import (
            row_count_above_mean,
            row_count_mean_crossing,
            row_iqr,
            row_mean,
            row_median,
            row_numba_max,
            row_numba_min,
            row_slope,
            row_std,
        )

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.n_instances_ <= 1:
            raise ValueError(
                "Supervised intervals requires more than 1 training time series."
            )

        if self.min_interval_length < 3:
            self._min_interval_length = 3

        if self.series_length_ < 6:
            raise ValueError("Series length must be at least 6.")
        elif self._min_interval_length * 2 > self.series_length_:
            raise ValueError(
                "Minimum interval length must be less than half the series length."
            )

        if self.features is None:
            self._features = [
                row_mean,
                row_median,
                row_std,
                row_slope,
                row_numba_min,
                row_numba_max,
                row_iqr,
                row_count_mean_crossing,
                row_count_above_mean,
            ]

        li = []
        if not isinstance(self._features, list):
            self._features = [self._features]

        for f in self._features:
            if callable(f):
                li.append(f)
            else:
                raise ValueError()
        self._features = li

        self._n_jobs = check_n_jobs(self.n_jobs)

        le = preprocessing.LabelEncoder()
        return le.fit_transform(y)

    def _generate_intervals(self, X, X_norm, y, idx, keep_transform):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        Xt = np.empty((self.n_instances_, 0)) if keep_transform else None
        intervals = []

        for i in range(self.n_dims_):
            for feature in self._features:
                random_cut_point = int(
                    rng.randint(
                        self._min_interval_length,
                        self.series_length_ - self._min_interval_length,
                    )
                )

                intervals_L, Xt_L = self._supervised_search(
                    X_norm[:, i, :random_cut_point],
                    y,
                    0,
                    feature,
                    i,
                    X[:, i, :],
                    rng,
                    keep_transform,
                )
                intervals.extend(intervals_L)
                if keep_transform:
                    Xt = np.hstack((Xt, Xt_L))

                intervals_R, Xt_R = self._supervised_search(
                    X_norm[:, i, random_cut_point:],
                    y,
                    random_cut_point,
                    feature,
                    i,
                    X[:, i, :],
                    rng,
                    keep_transform,
                )
                intervals.extend(intervals_R)
                if keep_transform:
                    Xt = np.hstack((Xt, Xt_R))

        return intervals, Xt

    def _transform_intervals(self, X, idx):
        if not self._transform_features[idx]:
            return np.zeros(X.shape[0])

        start, end, dim, feature = self.intervals_[idx]
        return feature(X[:, dim, start:end])

    def _supervised_search(
        self, X, y, ini_idx, feature, dim, X_ori, rng, keep_transform
    ):
        from sktime.utils.numba.stats import fisher_score

        intervals = []
        Xt = np.empty((X.shape[0], 0)) if keep_transform else None

        while X.shape[1] >= self._min_interval_length * 2:
            if (
                self.randomised_split_point
                and X.shape[1] != self._min_interval_length * 2
            ):
                div_point = rng.randint(
                    self._min_interval_length, X.shape[1] - self._min_interval_length
                )
            else:
                div_point = int(X.shape[1] / 2)

            sub_interval_0 = X[:, :div_point]
            sub_interval_1 = X[:, div_point:]

            interval_feature_0 = feature(sub_interval_0)
            interval_feature_1 = feature(sub_interval_1)

            score_0 = fisher_score(interval_feature_0, y)
            score_1 = fisher_score(interval_feature_1, y)

            if score_0 >= score_1 and score_0 != 0:
                end = ini_idx + len(sub_interval_0[0])

                intervals.append((ini_idx, end, dim, feature))
                X = sub_interval_0

                interval_feature_to_use = feature(X_ori[:, ini_idx:end])
                if keep_transform:
                    Xt = np.hstack(
                        (
                            Xt,
                            np.reshape(
                                interval_feature_to_use,
                                (interval_feature_to_use.shape[0], 1),
                            ),
                        )
                    )
            elif score_1 > score_0:
                ini_idx = ini_idx + div_point
                end = ini_idx + len(sub_interval_1[0])

                intervals.append((ini_idx, end, dim, feature))
                X = sub_interval_1

                interval_feature_to_use = feature(X_ori[:, ini_idx:end])
                if keep_transform:
                    Xt = np.hstack(
                        (
                            Xt,
                            np.reshape(
                                interval_feature_to_use,
                                (interval_feature_to_use.shape[0], 1),
                            ),
                        )
                    )
            else:
                break

        return intervals, Xt

    def set_features_to_transform(self, arr):
        """Set transform_features to the given array.

        Each index in the list corresponds to the index of an interval, True intervals
        are included in the transform, False intervals skipped and are set to 0.

        Parameters
        ----------
        arr : list of booleans of length len(self.intervals_)
             A list of intervals to skip.
        """
        if len(arr) != len(self.intervals_) or not all(
            isinstance(b, bool) for b in arr
        ):
            raise ValueError("Input must be a list bools of length len(intervals_).")
        self._transform_features = arr

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
        from sktime.utils.dependencies import _check_soft_dependencies

        params0 = {}

        if _check_soft_dependencies("numba", severity="none"):
            from sktime.utils.numba.stats import (
                row_mean,
                row_median,
                row_numba_max,
                row_numba_min,
            )

            params1 = {
                "n_intervals": 1,
                "features": [row_mean, row_numba_min, row_numba_max],
                "min_interval_length": 4,
            }
            params2 = {
                "n_intervals": 2,
                "randomised_split_point": False,
                "features": row_median,
            }
            return [params0, params1, params2]

        else:
            return params0
