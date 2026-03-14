"""Base classes for anomaly detectors.

    classes:
        BaseSegmentAnomalyDetector

By inheriting from these classes the remaining methods of the BaseDetector class to
implement to obtain a fully functional anomaly detector are given below.

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


class BaseSegmentAnomalyDetector(BaseDetector):
    """Base class for segment anomaly detectors.

    Segment anomaly detectors detect segments of data points that are considered
    anomalous.

    Output format of the `predict` method: See the `dense_to_sparse` method.
    Output format of the `transform` method: See the `sparse_to_dense` method.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
        "task": "segmentation",
    }

    @staticmethod
    def sparse_to_dense(
        y_sparse: pd.DataFrame, index: pd.Index, columns: pd.Index = None
    ) -> pd.DataFrame:
        """Convert the sparse output from the `predict` method to a dense format.

        Parameters
        ----------
        y_sparse : pd.DataFrame with RangeIndex
            Detected segment anomalies. Must have the following column:

            * ``"ilocs"`` -  left-closed intervals of iloc based segments.

            Can also have the following columns:

            * ``"icolumns"`` - array of identified variables for each anomaly.
        index : array-like
            Indices that are to be annotated according to `y_sparse`.
        columns : array-like
            Columns that are to be annotated according to `y_sparse`. Only relevant if
            y_sparse contains the column ``"icolumns"`` with identified variables.

        Returns
        -------
        pd.DataFrame with the input data index and one column:
            * ``"label"`` - integer labels ``1, ..., K`` for each segment anomaly.
            ``0`` is reserved for the normal instances.
        """
        if "icolumns" in y_sparse:
            return BaseSegmentAnomalyDetector._sparse_to_dense_icolumns(
                y_sparse, index, columns
            )
        return BaseSegmentAnomalyDetector._sparse_to_dense_ilocs(y_sparse, index)

    @staticmethod
    def dense_to_sparse(y_dense: pd.DataFrame) -> pd.DataFrame:
        """Convert the dense output from the `transform` method to a sparse format.

        Parameters
        ----------
        y_dense : pd.DataFrame
            The dense output from the `transform` method. It must either have the
            following column:

            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly and
            label 0 for normal instances.

            Or it must have columns of the form:

            * ``"labels_<*>"`` with integer labels ``1, ..., K`` for each segment
            anomaly, and 0 for normal instances.

        Returns
        -------
        pd.DataFrame :
            A ``pd.DataFrame`` with a range index and two columns:
            * ``"ilocs"`` - left-closed `pd.Interval`s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        ``output["ilocs"].array.left`` and ``output["ilocs"].array.right``,
        respectively.
        """
        if "labels" in y_dense.columns:
            return BaseSegmentAnomalyDetector._dense_to_sparse_ilocs(y_dense)
        elif y_dense.columns.str.startswith("labels_").all():
            return BaseSegmentAnomalyDetector._dense_to_sparse_icolumns(y_dense)
        raise ValueError(
            "Invalid columns in `y_dense`. Expected 'labels' or 'labels_*'."
            f" Got: {y_dense.columns}"
        )

    def _format_sparse_output(
        self,
        segment_anomalies: list[tuple[int, int]] | list[tuple[int, int, np.ndarray]],
        closed: str = "left",
    ) -> pd.DataFrame:
        """Format the sparse output of segment anomaly detectors.

        Can be reused by subclasses to format the output of the `_predict` method.

        Parameters
        ----------
        segment_anomalies : list
            List of tuples containing start and end indices of segment anomalies,
            and optionally a ``np.ndarray`` of the identified
            variables/components/columns.
        closed : str
            Whether the ``(start, end)`` tuple correspond to intervals that are closed
            on the left, right, both, or neither.

        Returns
        -------
        pd.DataFrame :
            A ``pd.DataFrame`` with a range index and two columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        ``output["ilocs"].array.left`` and ``output["ilocs"].array.right``,
        respectively.
        """
        # Cannot extract this from segment_anomalies as it may be an empty list.
        if self.get_tag("capability:variable_identification"):
            return self._format_sparse_output_icolumns(segment_anomalies, closed)
        else:
            return self._format_sparse_output_ilocs(segment_anomalies, closed)

    @staticmethod
    def _sparse_to_dense_ilocs(
        y_sparse: pd.DataFrame, index: pd.Index, columns: pd.Index = None
    ) -> pd.DataFrame:
        """Convert the sparse output from the `predict` method to a dense format.

        Parameters
        ----------
        y_sparse : pd.DataFrame with RangeIndex
            Detected segment anomalies. Must have the following column:

            * ``"ilocs"`` -  left-closed intervals of iloc based segments.
        index : array-like
            Indices that are to be annotated according to `y_sparse`.
        columns: array-like
            Not used. Only for API compatibility.

        Returns
        -------
        pd.DataFrame with the input data index and one column:
            * ``"label"`` - integer labels ``1, ..., K`` for each segment anomaly.
            ``0`` is reserved for the normal instances.
        """
        labels = pd.IntervalIndex(y_sparse["ilocs"]).get_indexer(index)
        # `get_indexer` return values 0 for the values inside the first interval, 1 to
        # the values within the next interval and so on, and -1 for values outside any
        # interval. The `skchange` convention is that 0 is normal and > 0 is anomalous,
        # so we add 1 to the result.
        labels += 1
        return pd.DataFrame(labels, index=index, columns=["labels"], dtype="int64")

    @staticmethod
    def _dense_to_sparse_ilocs(y_dense: pd.DataFrame) -> pd.DataFrame:
        """Convert the dense output from the `transform` method to a sparse format.

        Parameters
        ----------
        y_dense : pd.DataFrame
            The dense output from the `transform` method. Must have the following
            column:

            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly and
            label ``0`` for normal instances.

        Returns
        -------
        pd.DataFrame :
            A `pd.DataFrame` with a range index and two columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        ``output["ilocs"].array.left`` and ``output["ilocs"].array.right``,
        respectively.
        """
        # The sparse format only uses integer positions, so we reset the index.
        y_dense = y_dense["labels"].reset_index(drop=True)

        y_anomaly = y_dense.loc[y_dense.values > 0]
        anomaly_locations_diff = y_anomaly.index.diff()

        first_anomaly_start = y_anomaly.index[:1].to_numpy()
        anomaly_starts = y_anomaly.index[anomaly_locations_diff > 1]
        anomaly_starts = np.insert(anomaly_starts, 0, first_anomaly_start)

        last_anomaly_end = y_anomaly.index[-1:].to_numpy() + 1
        anomaly_ends = y_anomaly.index[np.roll(anomaly_locations_diff > 1, -1)] + 1
        anomaly_ends = np.insert(anomaly_ends, len(anomaly_ends), last_anomaly_end)

        anomaly_intervals = list(zip(anomaly_starts, anomaly_ends))
        return BaseSegmentAnomalyDetector._format_sparse_output_ilocs(
            anomaly_intervals, closed="left"
        )

    @staticmethod
    def _format_sparse_output_ilocs(
        anomaly_intervals: list[tuple[int, int]], closed: str = "left"
    ) -> pd.DataFrame:
        """Format the sparse output of segment anomaly detectors.

        Can be reused by subclasses to format the output of the `_predict` method.

        Parameters
        ----------
        anomaly_intervals : list
            List of tuples containing start and end indices of segment anomalies.

        Returns
        -------
        pd.DataFrame :
            A `pd.DataFrame` with a range index and two columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        ``output["ilocs"].array.left`` and ``output["ilocs"].array.right``,
        respectively.
        """
        anomaly_intervals = [(int(start), int(end)) for start, end in anomaly_intervals]
        return pd.DataFrame(
            {
                "ilocs": pd.IntervalIndex.from_tuples(anomaly_intervals, closed=closed),
                "labels": pd.RangeIndex(1, len(anomaly_intervals) + 1),
            },
        )

    @staticmethod
    def _sparse_to_dense_icolumns(
        y_sparse: pd.DataFrame, index: pd.Index, columns: pd.Index
    ) -> pd.DataFrame:
        """Convert the sparse output from the `predict` method to a dense format.

        Parameters
        ----------
        y_sparse : pd.DataFrame with RangeIndex
            Detected segment anomalies. Must have the following columns:

            * ``"ilocs"`` -  left-closed intervals of iloc based segments.
            * ``"icolumns"`` - array of identified variables for each anomaly.
        index : array-like
            Indices that are to be annotated according to `y_sparse`.
        columns : array-like
            Columns that are to be annotated according to `y_sparse`.

        Returns
        -------
        pd.DataFrame with the input data index and as many columns as in X:
            * ``"labels_<X.columns[i]>"`` for each column index i in ``X.columns``:
            Integer labels starting from ``0``.
        """
        anomaly_intervals = y_sparse["ilocs"].array
        anomaly_starts = anomaly_intervals.left
        anomaly_ends = anomaly_intervals.right
        anomaly_columns = y_sparse["icolumns"]

        start_is_open = anomaly_intervals.closed in ["neither", "right"]
        if start_is_open:
            anomaly_starts += 1  # Exclude the start index in the for loop below.
        end_is_closed = anomaly_intervals.closed in ["both", "right"]
        if end_is_closed:
            anomaly_ends += 1  # Include the end index in the for loop below.

        labels = np.zeros((len(index), len(columns)), dtype="int64")
        anomalies = zip(anomaly_starts, anomaly_ends, anomaly_columns)
        for i, (start, end, affected_columns) in enumerate(anomalies):
            labels[start:end, affected_columns] = i + 1

        prefixed_columns = [f"labels_{column}" for column in columns]
        return pd.DataFrame(labels, index=index, columns=prefixed_columns)

    @staticmethod
    def _dense_to_sparse_icolumns(y_dense: pd.DataFrame):
        """Convert the dense output from the `transform` method to a sparse format.

        Parameters
        ----------
        y_dense : pd.DataFrame
            The dense output from the `transform` method. Must have columns of the form:

            * `"labels_<*>"` with integer labels ``1, ..., K`` for each segment anomaly,
            and ``0`` for normal instances.

        Returns
        -------
        pd.DataFrame :
            A ``pd.DataFrame`` with a range index and three columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.
            * ``"icolumns"`` - list of affected columns for each anomaly.
        """
        # The sparse format only uses integer positions, so we reset index and columns.
        y_dense = y_dense.reset_index(drop=True)
        y_dense.columns = range(y_dense.columns.size)

        anomaly_intervals = []
        unique_labels = np.unique(y_dense.values)
        for i in unique_labels[unique_labels > 0]:
            anomaly_mask = y_dense == i
            which_columns = anomaly_mask.any(axis=0)
            which_rows = anomaly_mask.any(axis=1)
            anomaly_columns = anomaly_mask.columns[which_columns].to_list()
            anomaly_start = anomaly_mask.index[which_rows][0]
            anomaly_end = anomaly_mask.index[which_rows][-1]
            anomaly_intervals.append((anomaly_start, anomaly_end + 1, anomaly_columns))

        return BaseSegmentAnomalyDetector._format_sparse_output_icolumns(
            anomaly_intervals, closed="left"
        )

    @staticmethod
    def _format_sparse_output_icolumns(
        segment_anomalies: list[tuple[int, int, np.ndarray]],
        closed: str = "left",
    ) -> pd.DataFrame:
        """Format the sparse output of subset segment anomaly detectors.

        Can be reused by subclasses to format the output of the `_predict` method.

        Parameters
        ----------
        segment_anomalies : list
            List of tuples containing start and end indices of segment
            anomalies and a ``np.array`` of the affected components/columns.
        closed : str
            Whether the ``(start, end)`` tuple correspond to intervals that are closed
            on the left, right, both, or neither.

        Returns
        -------
        pd.DataFrame :
            A ``pd.DataFrame`` with a range index and three columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.
            * ``"icolumns"`` - list of affected columns for each anomaly.
        """
        ilocs = [(int(start), int(end)) for start, end, _ in segment_anomalies]
        icolumns = [
            np.array(components, dtype="int64")
            for _, _, components in segment_anomalies
        ]
        return pd.DataFrame(
            {
                "ilocs": pd.IntervalIndex.from_tuples(ilocs, closed=closed),
                "labels": pd.RangeIndex(1, len(ilocs) + 1),
                "icolumns": icolumns,
            }
        )
