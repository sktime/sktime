"""SoftED detection metric for soft/temporal-tolerant event evaluation.

Reference
---------
Salles, R., Escobar, L., Baroni, L., Zorrilla, R., Ziviani, A., Kreischer, V.,
Delicato, F., Pires, P. F., Maia, L., Coutinho, R., Assis, L., Ogasawara, E.
(2024). SoftED: Metrics for Soft Evaluation of Time Series Event Detection.
Computers & Industrial Engineering, 198, 109890.
https://doi.org/10.1016/j.cie.2024.109890
arXiv:2304.00439
"""

import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class SoftEDF1Score(BaseDetectionMetric):
    """SoftED F1 score for event detection with temporal tolerance.

    Computes a soft F1 score between predicted and true event iloc positions.
    Each prediction's contribution is a soft-membership value in ``[0, 1]``
    based on its distance to the nearest unmatched true event, within a
    tolerance window. A prediction farther than ``tolerance`` iloc units
    from every true event contributes zero.

    For true event ``t_i`` and predicted event ``p_j`` with
    ``d = |t_i - p_j|`` and ``T = tolerance``:

    * ``membership="linear"``: ``mu(d) = max(0, 1 - d / T)``
    * ``membership="rectangular"``: ``mu(d) = 1 if d <= T else 0``

    Soft precision and recall follow Salles et al. (2024) via greedy
    one-to-one matching of each prediction to its nearest unused true event
    within tolerance. F1 is the harmonic mean of soft precision and recall.

    Parameters
    ----------
    tolerance : int, optional (default=5)
        Half-width of the soft-membership window, in iloc units.
        Must be a non-negative integer.
    membership : {"linear", "rectangular"}, optional (default="linear")
        Soft-membership function mapping distance to a score in ``[0, 1]``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import SoftEDF1Score
    >>> y_true = pd.DataFrame({"ilocs": [10, 50, 90]})
    >>> y_pred = pd.DataFrame({"ilocs": [12, 51, 200]})
    >>> metric = SoftEDF1Score(tolerance=5, membership="linear")
    >>> score = metric(y_true, y_pred)
    >>> 0.0 < score <= 1.0
    True
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,
    }

    def __init__(self, tolerance=5, membership="linear"):
        self.tolerance = tolerance
        self.membership = membership
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute SoftED F1 between true and predicted events.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground-truth events in "points" format. Must have column ``ilocs``.
        y_pred : pd.DataFrame
            Predicted events in "points" format. Must have column ``ilocs``.
        X : pd.DataFrame, optional (default=None)
            Unused; kept for interface parity.

        Returns
        -------
        float
            Soft F1 score in ``[0, 1]``. Higher is better.
        """
        gt = np.sort(np.asarray(y_true["ilocs"].values, dtype=float))
        pred = np.sort(np.asarray(y_pred["ilocs"].values, dtype=float))

        if len(gt) == 0 and len(pred) == 0:
            return 1.0
        if len(gt) == 0 or len(pred) == 0:
            return 0.0

        tol = float(self.tolerance)
        if tol < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance!r}")

        membership = self.membership
        if membership not in ("linear", "rectangular"):
            raise ValueError(
                f"membership must be 'linear' or 'rectangular', got {membership!r}"
            )

        # Greedy one-to-one match each prediction to its nearest unused true
        # event within tolerance. Accumulate soft-membership contributions.
        used = np.zeros(len(gt), dtype=bool)
        soft_matches = 0.0
        for p in pred:
            d_all = np.abs(gt - p)
            d_all[used] = np.inf
            j = int(np.argmin(d_all))
            d = float(d_all[j])
            if d > tol:
                continue
            if membership == "linear":
                soft_matches += max(0.0, 1.0 - d / tol) if tol > 0 else float(d == 0)
            else:  # rectangular
                soft_matches += 1.0
            used[j] = True

        soft_precision = soft_matches / len(pred)
        soft_recall = soft_matches / len(gt)

        if soft_precision + soft_recall == 0:
            return 0.0
        return 2.0 * soft_precision * soft_recall / (soft_precision + soft_recall)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return. Unused here; kept
            for interface parity with the sktime test framework.

        Returns
        -------
        list of dict
            Parameter dicts for automated test-instance construction.
        """
        return [
            {},
            {"tolerance": 10, "membership": "linear"},
            {"tolerance": 3, "membership": "rectangular"},
        ]
