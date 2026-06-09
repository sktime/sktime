"""Base classes for change point detectors.

    classes:
        BaseChangeDetector

By inheriting from these classes the remaining methods of the BaseDetector class to
implement to obtain a fully functional change point detector are given below.

Needs to be implemented:
    _fit(self, X, y=None)
    _predict(self, X)

Optional to implement:
    _transform_scores(self, X)
    _update(self, X, y=None)
"""

import numpy as np
import pandas as pd

from ..base import BaseDetector


class BaseChangeDetector(BaseDetector):
    """Base class for change detectors.

    Changepoint detectors detect points in time where a change in the data occurs.
    Data between two change points is a segment where the data is considered to be
    homogeneous, i.e., of the same distribution. A change point is defined as the
    location of the first element of a segment.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
        "task": "change_point_detection",
    }

    @staticmethod
    def sparse_to_dense(
        y_sparse: pd.DataFrame, index: pd.Index, columns: pd.Index = None
    ) -> pd.Series:
        """Convert the sparse output from the `predict` method to a dense format.

        Parameters
        ----------
        y_sparse : pd.DataFrame
            The sparse output from a change point detector's `predict` method.
        index : array-like
            Indices that are to be annotated according to `y_sparse`.
        columns: array-like
            Not used. Only for API compatibility.

        Returns
        -------
        pd.DataFrame with the input data index and one column:
        * ``"label"`` - integer labels ``0, ..., K`` for each segment between two
        change points.
        """
        changepoints = y_sparse["ilocs"].to_list()
        n = len(index)
        changepoints = [0] + changepoints + [n]
        segment_labels = np.zeros(n)
        for i in range(len(changepoints) - 1):
            segment_labels[changepoints[i] : changepoints[i + 1]] = i

        return pd.DataFrame(
            segment_labels, index=index, columns=["labels"], dtype="int64"
        )

    @staticmethod
    def dense_to_sparse(y_dense: pd.DataFrame) -> pd.DataFrame:
        """Convert the dense output from the `transform` method to a sparse format.

        Parameters
        ----------
        y_dense : pd.DataFrame
            The dense output from a change point detector's `transform` method.

        Returns
        -------
        pd.DataFrame :
            A `pd.DataFrame` with a range index and one column:
            * ``"ilocs"`` - integer locations of the change points.
        """
        is_changepoint = y_dense["labels"].diff().abs() > 0
        changepoints = y_dense.index[is_changepoint]
        return BaseChangeDetector._format_sparse_output_ilocs(changepoints)

    @staticmethod
    def _format_sparse_output_ilocs(changepoints) -> pd.DataFrame:
        """Format the sparse output of change point detectors.

        Can be reused by subclasses to format the output of the `_predict` method.

        Parameters
        ----------
        changepoints : list
            List of change point locations.

        Returns
        -------
        pd.DataFrame :
            A `pd.DataFrame` with a range index and one column:
            * ``"ilocs"`` - integer locations of the change points.
        """
        return pd.DataFrame(changepoints, columns=["ilocs"], dtype="int64")

    def _format_sparse_output(self, changepoints) -> pd.DataFrame:
        """Format the sparse output of change point detectors.

        Can be reused by subclasses to format the output of the `_predict` method.

        Parameters
        ----------
        changepoints : list
            List of changepoint locations.

        Returns
        -------
        pd.DataFrame :
            A `pd.DataFrame` with a range index and one column:
            * ``"ilocs"`` - integer locations of the change points.
        """
        return self._format_sparse_output_ilocs(changepoints)
