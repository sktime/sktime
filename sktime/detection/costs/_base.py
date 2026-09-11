# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten, johannvk
"""Base cost function for interval-based scoring."""

__author__ = ["Tveten", "johannvk"]


from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


class BaseCost(BaseIntervalScorer):
    """Stateless base class for cost functions.

    Costs evaluate the fit of a model on data intervals. This base class
    provides the stateless API where ``evaluate(X, cuts)`` receives data at
    call time — no ``fit()`` step is needed and no data is stored.

    If the cost supports fixed parameters, that is indicated by the
    ``supports_fixed_param`` tag.

    Parameters
    ----------
    param : None, optional (default=None)
        If ``None``, the cost is evaluated with parameters that minimize the cost.
        If ``param`` is not ``None``, the cost is evaluated at that fixed parameter.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "task": "cost",
        "supports_fixed_param": False,
    }

    def __init__(self, param=None):
        self.param = param
        super().__init__()
        if self.param is not None and not self.get_tag("supports_fixed_param"):
            raise ValueError(
                "This cost does not support fixed parameters."
                f" Got {type(self).__name__} with param={self.param}."
            )

    def _check_param(self, param, X):
        """Check the parameter with respect to the input data."""
        if param is None:
            return None
        return self._check_fixed_param(param, X)

    def _check_fixed_param(self, param, X):
        """Check the fixed parameter with respect to the input data.

        Override in subclasses for specific validation.
        """
        return param

    def _evaluate(self, X, cuts):
        """Evaluate the cost on data intervals.

        Parameters
        ----------
        X : np.ndarray
            2D data array.
        cuts : np.ndarray
            2D array with two columns ``[start, end]`` per row.

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row per interval.
        """
        starts, ends = cuts[:, 0], cuts[:, 1]
        if self.param is None:
            return self._evaluate_optim_param(X, starts, ends)
        else:
            param = self._check_param(self.param, X)
            return self._evaluate_fixed_param(X, starts, ends, param)

    def _evaluate_optim_param(self, X, starts, ends):
        """Evaluate cost with optimal (MLE) parameters.

        Parameters
        ----------
        X : np.ndarray
            2D data array.
        starts : np.ndarray
            Start indices of intervals (inclusive).
        ends : np.ndarray
            End indices of intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs.
        """
        raise NotImplementedError("abstract method")

    def _evaluate_fixed_param(self, X, starts, ends, param):
        """Evaluate cost with a fixed parameter.

        Parameters
        ----------
        X : np.ndarray
            2D data array.
        starts : np.ndarray
            Start indices of intervals (inclusive).
        ends : np.ndarray
            End indices of intervals (exclusive).
        param : object
            The checked fixed parameter.

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs.
        """
        raise NotImplementedError("abstract method")
