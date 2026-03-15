"""Interval scorer base class.

    class name: BaseIntervalScorer

Scitype defining methods:
    fitting                         - fit(self, X, y=None)
    evaluating                      - evaluate(self, cuts)

Needs to be implemented for a concrete interval scorer:
    _fit(self, X, y=None)
    _evaluate(self, cuts)

Recommended but optional to implement for a concrete detector:
    min_size(self)
    get_model_size(self, p)
"""

__author__ = ["Tveten", "johannvk", "fkiraly"]
__all__ = ["BaseIntervalScorer"]

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sktime.base import BaseEstimator
from sktime.utils.validation.series import check_series

from sktime.detection._skchange.utils.validation.cuts import check_cuts_array
from sktime.detection._skchange.utils.validation.data import as_2d_array


class BaseIntervalScorer(BaseEstimator):
    """Base class template for interval scorers.

    This is a common base class for costs, change scores, and anomaly scores. It is used
    as a building block for detectors in skchange. The class provides a common interface
    to evaluate a scoring function on a set of data cuts. Depending on the sub class,
    the cuts may represent either intervals to subset or splits within intervals.

    Attributes
    ----------
    _is_fitted : bool
        Indicates whether the interval scorer has been fitted.
    _X : np.ndarray
        The data input to `fit` coerced to a 2D ``np.ndarray``.
    _required_cut_size : int
        The required size of the cuts array, determined by the task of the scorer.
    """

    _tags = {
        "object_type": "interval_scorer",  # type of object
        "authors": ["Tveten", "johannvk", "fkiraly"],  # author(s) of the object
        "maintainers": "Tveten",  # current maintainer(s) of the object
        "task": None,  # "cost", "change_score", "local_anomaly_score", "saving"
        "distribution_type": "None",  # "None", "Poisson", "Gaussian"
        # is_conditional: whether the scorer uses some of the input variables as
        # covariates in a regression model or similar. If `True`, the scorer requires
        # at least two input variables. If `False`, all p input variables/columns are
        # used to evaluate the score, such that the output has either 1 or p columns.
        "is_conditional": False,
        # is_aggregated: whether the scorer always returns a single value per cut or
        # not, irrespective of the input data shape.
        # Many scorers will not be aggregated, for example all scorers that evaluate
        # each input variable separately and return a score vector with one score for
        # each variable.
        "is_aggregated": False,
        # is_penalised: indicates whether the score is inherently penalised (True) or
        # not (False). If `True`, a score > 0 means that a change or anomaly is
        # detected. Penalised scores can be both positive and negative.
        # If `False`, the score is not penalised. To test for the existence of a change,
        # penalisation must be performed externally. Such scores are always
        # non-negative.
        "is_penalised": False,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:update": False,
    }

    def __init__(self):
        self._is_fitted = False
        self._X = None
        self._required_cut_size = None

        super().__init__()

    def fit(self, X, y=None):
        """Fit the interval scorer to the training data.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or np.ndarray
            Data to evaluate.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates the fitted model and sets attributes ending in ``"_"``.
        """
        # Getting the cut size per call to evaluate generates a surprising amount of
        # overhead, so we cache it here.
        # Cannot be done in __init__ because the task is set at the very end of
        # __init__ in subclasses.
        self._required_cut_size = self._get_required_cut_size()

        X = check_series(X, allow_index_names=True)
        if isinstance(X, pd.DataFrame):
            self._X_columns = X.columns
        else:
            self._X_columns = None
        self._X = as_2d_array(X)

        self._fit(X=self._X, y=y)
        self._is_fitted = True

        return self

    def _fit(self, X: np.ndarray, y=None):
        """Fit the interval scorer to training data.

        The core logic of fitting an interval scorer to training data is implemented
        here.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in ``"_"``.
        """
        return self

    def evaluate(self, cuts: ArrayLike) -> np.ndarray:
        """Evaluate the score according to a set of cuts.

        Parameters
        ----------
        cuts : ArrayLike
            A 2D array of integer location-based cuts to evaluate where each row gives
            a single cut specification. If a 1D array is passed, it is assumed to
            be a row vector. Each cut divide the data into one or more intervals for
            evaluation and may contain multiple entries representing for example the
            start and end of the interval, and potentially split points within the
            interval. Each cut must be sorted in increasing order. Subclasses specify
            the further expected structure of the cuts array and how it is used
            internally to evaluate the score.

        Returns
        -------
        scores : np.ndarray
            A 2D array of scores. One row for each row in cuts.
        """
        self.check_is_fitted()
        cuts = as_2d_array(cuts, vector_as_column=False)
        cuts = self._check_cuts(cuts)
        values = self._evaluate(cuts)
        return values

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the score on a set of cuts.

        The core logic of evaluating a function on according to the cuts is implemented
        here.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array of integer location-based cuts to evaluate. Each row in the array
            must be sorted in increasing order.

        Returns
        -------
        values : np.ndarray
            A 2D array of scores. One row for each row in cuts.
        """
        raise NotImplementedError("abstract method")

    @property
    def min_size(self) -> int | None:
        """Minimum valid size of an interval to evaluate.

        The size of each interval is by default defined as ``np.diff(cuts[i, ])``.
        Subclasses can override the min_size to mean something else, for example in
        cases where intervals are combined before evaluation or `cuts` specify
        disjoint intervals.

        Returns
        -------
        int or None
            The minimum valid size of an interval to evaluate. If ``None``, it is
            unknown what the minimum size is. E.g., the scorer may need to be fitted
            first to determine the minimum size.
        """
        return 1

    def get_model_size(self, p: int) -> int:
        """Get the number of model parameters to estimate for each interval.

        The primary use of this method is to determine an appropriate default penalty
        value in detectors. For example, a scorer for a change in mean has one
        parameter to estimate per variable in the data, a scorer for a change in the
        mean and variance has two parameters to estimate per variable, and so on.
        Subclasses should override this method accordingly.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return p

    def _get_required_cut_size(self) -> int:
        """Get the required cut size for the scorer.

        The cut size is the number of columns in the cuts array. The cut size is
        determined by the task of the scorer. For example, a cost and a saving has a cut
        size of 2, a change score has a cut size of 3, and a local anomaly score has a
        cut size of 4. The cut size is used to check the cuts array for compatibility.
        """
        task = self.get_tag("task")
        if task == "cost":
            cut_size = 2
        elif task == "change_score":
            cut_size = 3
        elif task == "saving":
            cut_size = 2
        elif task == "local_anomaly_score":
            cut_size = 4
        else:
            raise RuntimeError("The task of the interval scorer is not set.")

        return cut_size

    def _check_cuts(self, cuts: np.ndarray) -> np.ndarray:
        """Check cuts for compatibility.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array of integer location-based cuts to evaluate. Each row in the array
            must be sorted in increasing order.

        Returns
        -------
        cuts : np.ndarray
            The unmodified input `cuts` array.

        Raises
        ------
        ValueError
            If the `cuts` are not compatible.
        """
        return check_cuts_array(
            cuts,
            n_samples=self._X.shape[0],
            min_size=self.min_size,
            last_dim_size=self._required_cut_size,
        )

    def check_is_penalised(self):
        """Check if the scorer is inherently performing penalisation."""
        if not self.get_tag("is_penalised"):
            raise RuntimeError("The interval scorer is not penalised.")

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the input data."""
        self.check_is_fitted()
        return self._X.shape[0]

    @property
    def n_variables(self) -> int:
        """Return the number of variables in the input data."""
        self.check_is_fitted()
        return self._X.shape[1]
