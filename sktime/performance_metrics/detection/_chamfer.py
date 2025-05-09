"""Directed Chamfer distance between two sets of points."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection.utils import _find_closest_elements


class DirectedChamfer(BaseDetectionMetric):
    r"""Directed Chamfer distance between event points.

    For detected time points :math:`A = (a_1, a_2, \ldots, a_n)` and true time points
    :math:`B = (b_1, b_2, \ldots, b_m)`,
    the directed (unnormalized) Chamfer distance is defined as:

    .. math::

        d(A, B) = \sum_{i=1}^{n} \left| a_i - b'_i \right|

    where :math:`b'_i` is the true event closest to :math:`a_i`,
    that is, :math:`b'_i = \arg \min_{b\in B} |a_i - b|`.

    If ``X`` is provided, the time points are taken as the location indices in ``X``.
    Otherwise, it is assumed that ``X`` has a ``RangeIndex``.

    Parameters
    ----------
    normalize : bool, default=False
        If True, the Chamfer distance is normalized by the number of detected events,
        i.e., divided by :math:`n`.
        If unnormalized, making too many detections will be penalized, wherease
        the normalized distance penalizes only the average distance to the
        closest true events, regardless of the number of detections.
    """

    _tags = {
        "scitype:y": "points",  # or segments
        "requires_X": False,
        "lower_is_better": True,
    }

    def __init__(self, normalize=False):
        self.normalize = normalize

        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) event locations, in ``X``.
            Should be ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D),
            of ``Series`` scitype = individual time series.

            For further details on data format, see glossary on :term:`mtype`.

        y_pred : time series in ``sktime`` compatible data container format
            Detected events to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        X : optional, pd.DataFrame, pd.Series or np.ndarray
            Time series that is being labelled.
            If not provided, assumes ``RangeIndex`` for ``X``, and that
            values in ``X`` do not matter.

        Returns
        -------
        loss : float
            Calculated metric.
        """
        y_true_ilocs = y_true.ilocs
        y_pred_ilocs = y_pred.ilocs

        if X is not None and not isinstance(X.index, pd.RangeIndex):
            y_true_locs = X.index[y_true_ilocs]
            y_pred_locs = X.index[y_pred_ilocs]
        else:
            y_true_locs = y_true_ilocs
            y_pred_locs = y_pred_ilocs

        y_true_locs = y_true_locs.to_numpy()
        y_pred_locs = y_pred_locs.to_numpy()

        y_true_closest = _find_closest_elements(y_pred_locs, y_true_locs)
        y_true_closest = np.array(y_true_closest)

        distance = np.sum(np.abs(y_true_closest - y_pred_locs))

        if self.normalize:
            distance /= len(y_pred_locs)

        return distance

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        param1 = {}
        param2 = {"normalize": True}

        return [param1, param2]
