# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten, johannvk, fkiraly
"""Stateless base class for interval scorers.

    class name: BaseIntervalScorer

Unlike the original skchange version, this base class is **stateless**:
- ``evaluate(X, cuts)`` accepts data at call time — no ``fit()`` step needed.
- No ``self._X`` storage — eliminates memory leaks when scorers are stored
  as hyperparameters on detector instances.

Subclasses must implement:
    ``_evaluate(X, starts, ends, ...)`` — core scoring logic.
"""

__author__ = ["Tveten", "johannvk", "fkiraly"]
__all__ = ["BaseIntervalScorer"]


from sktime.base import BaseEstimator
from sktime.detection._utils import as_2d_array, check_cuts_array


class BaseIntervalScorer(BaseEstimator):
    """Stateless base class for interval scorers.

    This is a common base class for costs, change scores, and anomaly scores
    used as building blocks for detection algorithms. The class provides a
    common interface to evaluate a scoring function on data and cuts.

    **Key difference from skchange's original design**: This class is
    **stateless**. The ``evaluate(X, cuts)`` method receives data directly
    at evaluation time. There is no ``fit()`` step, and no ``self._X``
    attribute. This prevents memory leaks when scorers are stored as
    hyperparameters on detector instances (e.g., ``PELT(cost=L2Cost())``).

    Subclasses implement ``_evaluate(X, cuts)`` where ``X`` is a 2D numpy
    array and ``cuts`` is a 2D integer array of indices.

    Attributes
    ----------
    _required_cut_size : int
        Required number of columns in the cuts array, determined by task.
    """

    _tags = {
        "object_type": "interval_scorer",
        "authors": ["Tveten", "johannvk", "fkiraly"],
        "maintainers": "Tveten",
        "task": None,  # "cost", "change_score", "local_anomaly_score", "saving"
        "distribution_type": "None",
        "is_conditional": False,
        "is_aggregated": False,
        "is_penalised": False,
        "capability:multivariate": True,
        "capability:missing_values": False,
    }

    def __init__(self):
        super().__init__()

    def evaluate(self, X, cuts):
        """Evaluate the score according to data and a set of cuts.

        This is stateless: X is passed at call time, not stored.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or np.ndarray
            Data to score. Will be coerced to a 2D numpy array.
        cuts : array-like
            A 2D array of integer location-based cuts. Each row gives a single
            cut specification. If a 1D array is passed, it is treated as a
            single row. The structure depends on the task:

            - cost (cut_size=2): ``[start, end]``
            - change_score (cut_size=3): ``[start, split, end]``
            - saving (cut_size=2): ``[start, end]``
            - local_anomaly_score (cut_size=4): ``[start, inner_start, inner_end, end]``

        Returns
        -------
        scores : np.ndarray
            A 2D array of scores. One row for each row in cuts.
        """
        X = as_2d_array(X)
        cuts = as_2d_array(cuts, vector_as_column=False)

        required_cut_size = self._get_required_cut_size()
        cuts = check_cuts_array(
            cuts,
            n_samples=X.shape[0],
            min_size=self.min_size,
            last_dim_size=required_cut_size,
        )
        return self._evaluate(X, cuts)

    def _evaluate(self, X, cuts):
        """Evaluate the score on data and cuts.

        Core logic implemented by subclasses.

        Parameters
        ----------
        X : np.ndarray
            2D data array.
        cuts : np.ndarray
            2D integer array of cuts.

        Returns
        -------
        values : np.ndarray
            A 2D array of scores. One row for each row in cuts.
        """
        raise NotImplementedError("abstract method")

    @property
    def min_size(self):
        """Minimum valid size of an interval to evaluate.

        Returns
        -------
        int or None
        """
        return 1

    def get_model_size(self, p):
        """Get the number of model parameters per interval.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
        """
        return p

    def _get_required_cut_size(self):
        """Get the required cut size for the scorer.

        Returns
        -------
        int
            Number of columns expected in the cuts array.
        """
        task = self.get_tag("task")
        if task == "cost":
            return 2
        elif task == "change_score":
            return 3
        elif task == "saving":
            return 2
        elif task == "local_anomaly_score":
            return 4
        else:
            raise RuntimeError(
                f"The task of the interval scorer is not set or unrecognized: {task}"
            )

    def check_is_penalised(self):
        """Check if the scorer is inherently performing penalisation."""
        if not self.get_tag("is_penalised"):
            raise RuntimeError("The interval scorer is not penalised.")
