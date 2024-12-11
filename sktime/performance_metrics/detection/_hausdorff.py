"""Directed Hausdorff distance between two sets of points."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class DirectedHausdorff(BaseDetectionMetric):
    """Directed Hausdorff metric between event points.

    
    """

    _tags = {
        "scitype:y": "points",  # or segments
        "requires_X": False,
        "lower_is_better": True,
    }

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

        y_true_closest = self._find_closest_elements(y_pred_locs, y_true_locs)
        y_true_closest = np.array(y_true_closest)

        distance = np.sum(np.abs(y_true_closest - y_pred_locs))
        return distance

    def _find_closest_elements(self, a, b):
        """Find the closest element in b for each element in a.

        Parameters
        ----------
        a : 1D array-like
            An ordered (sorted) list of elements.
        b : 1D array-like
            Another ordered (sorted) list of elements.

        Returns
        -------
        closest : list, same length as a
            a list of closest elements in ``b`` for each element in ``a``.
            In case of ties, the first closest element is chosen.

        Examples
        --------
        >>> a = [1, 3, 5]
        >>> b = [2, 3.1, 3.2, 4, 6]
        >>> pointer = DirectedHausdorff()._find_closest_elements(a, b)
        """
        # Pointers for traversing A and B
        i, j = 0, 0
        n, m = len(a), len(b)

        # List to store the result
        result = []

        while i < n:
            # Move pointer j in b to get the closest value to a[i]
            while j + 1 < m and abs(b[j + 1] - a[i]) < abs(b[j] - a[i]):
                j += 1

            # Append the closest value in b for a[i]
            result.append(b[j])

            # Move to the next element in a
            i += 1

        return result
