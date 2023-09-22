"""Random interval features.

A transformer for the extraction of features on randomly selected intervals.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["RandomIntervals"]

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.base._base import _clone_estimator
from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.summarize import SummaryTransformer


class RandomIntervals(BaseTransformer):
    """Random interval feature transformer.

    Extracts intervals with random length, position and dimension from series in fit.
    Transforms each interval subseries using the given transformer(s) and concatenates
    them into a feature vector in transform.

    Currently, the transform is re-fit for every interval in transform. As such, it may
    not be suitable for some supervised transformers in its current state.

    Parameters
    ----------
    n_intervals : int, default=100,
        The number of intervals of random length, position and dimension to be
        extracted.
    transformers : transformer or list of transformers, default=None,
        Transformer(s) used to extract features from each interval. If None, defaults to
        the SummaryTransformer using
        [mean, median, min, max, std, 25% quantile, 75% quantile]
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    See Also
    --------
    SupervisedIntervals
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "capability:unequal_length": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        n_intervals=100,
        transformers=None,
        random_state=None,
        n_jobs=1,
    ):
        self.n_intervals = n_intervals
        self.transformers = transformers

        self.random_state = random_state
        self.n_jobs = n_jobs

        self._transformers = transformers
        self._intervals = []
        self._dims = []

        super().__init__()

    def _fit(self, X, y=None):
        """Fit the random interval transform.

        Parameters
        ----------
        X : pandas DataFrame or 3d numpy array, input time series
        y : array_like, target values (optional, ignored)
        """
        _, n_dims, series_length = X.shape

        if self.transformers is None:
            self._transformers = [
                SummaryTransformer(
                    summary_function=("mean", "std", "min", "max"),
                    quantiles=(0.25, 0.5, 0.75),
                )
            ]

        if not isinstance(self._transformers, list):
            self._transformers = [self._transformers]

        li = []
        for i in range(len(self._transformers)):
            li.append(
                _clone_estimator(
                    self._transformers[i],
                    self.random_state,
                )
            )

            m = getattr(li[i], "n_jobs", None)
            if m is not None:
                li[i].n_jobs = self.n_jobs
        self._transformers = li

        rng = check_random_state(self.random_state)
        self._dims = rng.choice(n_dims, self.n_intervals, replace=True)
        self._intervals = np.zeros((self.n_intervals, 2), dtype=int)

        for i in range(0, self.n_intervals):
            if rng.random() < 0.5:
                self._intervals[i][0] = rng.randint(0, series_length - 3)
                length = (
                    rng.randint(0, series_length - self._intervals[i][0] - 3) + 3
                    if series_length - self._intervals[i][0] - 3 > 0
                    else 3
                )
                self._intervals[i][1] = self._intervals[i][0] + length
            else:
                self._intervals[i][1] = rng.randint(0, series_length - 3) + 3
                length = (
                    rng.randint(0, self._intervals[i][1] - 3) + 3
                    if self._intervals[i][1] - 3 > 0
                    else 3
                )
                self._intervals[i][0] = self._intervals[i][1] - length
        return self

    def _transform(self, X, y=None):
        """Transform data into random interval features.

        Parameters
        ----------
        X : pandas DataFrame or 3d numpy array, input time series
        y : array_like, target values (optional, ignored)

        Returns
        -------
        Pandas dataframe of random interval features.
        """
        X_t = []
        for i in range(0, self.n_intervals):
            for j in range(len(self._transformers)):
                t = self._transformers[j].fit_transform(
                    np.expand_dims(
                        X[
                            :,
                            self._dims[i],
                            self._intervals[i][0] : self._intervals[i][1],
                        ],
                        axis=1,
                    ),
                    y,
                )

                if isinstance(t, pd.DataFrame):
                    t = t.to_numpy()

                if i == 0 and j == 0:
                    X_t = t
                else:
                    X_t = np.concatenate((X_t, t), axis=1)

        return pd.DataFrame(X_t)

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
        return {"n_intervals": 3}
