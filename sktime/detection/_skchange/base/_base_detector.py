"""Detector base class.

    class name: BaseDetector

    Adapted from the BaseDetector class in sktime.

Scitype defining methods:
    fitting                         - fit(self, X, y=None)
    detecting, sparse format        - predict(self, X)
    detecting, dense format         - transform(self, X)
    detection scores, dense         - transform_scores(self, X)  [optional]
    updating (temporal)             - update(self, X, y=None)  [optional]

Each detector type (e.g. point anomaly detector, segment anomaly detector,
changepoint detector) are subclasses of BaseDetector (task tag in sktime).
A detector type is defined by the content and format of the output of the predict
method. Each detector type therefore has the following methods for converting between
sparse and dense output formats:
    converting sparse output to dense - sparse_to_dense(y_sparse, index, columns)
    converting dense output to sparse - dense_to_sparse(y_dense)  [optional]

Convenience methods:
    update&detect   - update_predict(self, X)
    fit&detect      - fit_predict(self, X, y=None)
    fit&transform   - fit_transform(self, X, y=None)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()

Needs to be implemented for a concrete detector:
    _fit(self, X, y=None)
    _predict(self, X)
    sparse_to_dense(y_sparse, index)  - implemented by sub base classes in skchange

Recommended but optional to implement for a concrete detector:
    dense_to_sparse(y_dense)
    _transform_scores(self, X)
    _update(self, X, y=None)
"""

__author__ = ["Tveten"]
__all__ = ["BaseDetector"]

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector as _BaseDetector


class BaseDetector(_BaseDetector):
    """Base class for all detectors in skchange.

    Adjusts the BaseDetector class in sktime to fit the skchange framework as follows:

    * Sets reasonable default values for the tags.

    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Tveten"],  # author(s) of the object
        "maintainers": ["Tveten"],  # current maintainer(s) of the object
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # str or list of str, package soft dependencies
        # estimator tags
        # --------------
        "object_type": "detector",  # type of object
        "learning_type": "unsupervised",  # supervised, unsupervised
        "task": "None",  # anomaly_detection, change_point_detection, segmentation
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:update": False,
        "capability:variable_identification": False,
        "distribution_type": "None",
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
    }

    def transform(self, X) -> pd.DataFrame:
        """Create labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with same index as X
            Labels for sequence `X`.

            * If ``task`` is ``"anomaly_detection"``, the values are integer labels.
              A value of 0 indicates that `X`, at the same time index, has no anomaly.
              Other values indicate an anomaly.
              Most detectors will return 0 or 1, but some may return more values,
              if they can detect different types of anomalies.
              indicating whether `X`, at the same
              index, is an anomaly, 0 for no, 1 for yes.
            * If ``task`` is ``"changepoint_detection"``, the values are integer labels,
              indicating labels for segments between changepoints.
              Possible labels are integers starting from 0.
            * If ``task`` is "segmentation", the values are integer labels of the
              segments. Possible labels are integers starting from 0.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        columns = X.columns
        index = X.index

        y_sparse = self.predict(X)
        y_dense = self.sparse_to_dense(y_sparse, range(len(X)), columns)

        y_dense.index = index
        return y_dense

    def fit_transform(self, X, y=None) -> pd.DataFrame:
        """Fit to data, then transform it.

        Fits model to `X` and `y` with given detection parameters
        and returns the detection labels made by the model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be predicted.

        Returns
        -------
        y : pd.DataFrame with same index as X
            Labels for sequence `X`.

            * If ``task`` is ``"anomaly_detection"``, the values are integer labels.
              A value of 0 indicatesthat `X`, at the same time index, has no anomaly.
              Other values indicate an anomaly.
              Most detectors will return 0 or 1, but some may return more values,
              if they can detect different types of anomalies.
              indicating whether `X`, at the same
              index, is an anomaly, 0 for no, 1 for yes.
            * If ``task`` is ``"changepoint_detection"``, the values are integer labels,
              indicating labels for segments between changepoints.
              Possible labels are integers starting from 0.
            * If ``task`` is "segmentation", the values are integer labels of the
              segments. Possible labels are integers starting from 0.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        columns = X.columns
        index = X.index

        y_sparse = self.fit_predict(X)
        y_dense = self.sparse_to_dense(y_sparse, range(len(X)), columns)

        y_dense.index = index
        return y_dense

    def predict_segments(self, X) -> pd.DataFrame:
        """Predict segments on test/deployment data.

        The main difference to `predict` is that this method always returns
        a ``pd.DataFrame`` with segments of interest, even if the task is not
        segmentation.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with RangeIndex

            ``pd.DataFrame`` with the following columns:

            * ``"ilocs"`` - always. Values are left-closed intervals with
              left/right values being ``iloc`` references to indices of `X`,
              signifying segments.
            * ``"labels"`` - if the task, by tags, is supervised or semi-supervised
              segmentation, or segment clustering.

            The meaning of segments in the ``"ilocs"`` column and ``"labels"``
            column is as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the intervals are intervals between changepoints/anomalies, and
              potential labels are consecutive integers starting from 0.
            * If ``task`` is ``"segmentation"``, the values are segmentation labels.
        """
        self.check_is_fitted()

        task = self.get_tag("task")
        if task in ["anomaly_detection", "change_point_detection"]:
            y_pred_pts = self.predict_points(X)
            y_pred = self.change_points_to_segments(y_pred_pts, start=0, end=len(X))
        elif task == "segmentation":
            y_pred = self._predict_segments(X)

        return y_pred

    @staticmethod
    def change_points_to_segments(y_sparse, start=None, end=None) -> pd.DataFrame:
        """Convert an series of change point indexes to segments.

        Parameters
        ----------
        y_sparse : pd.DataFrame with RangeIndex
            Detected change points. Must have the following column:

            * ``"ilocs"`` - the iloc indices at which the change points take place.
            Sorted in ascending order.
        start : optional, default=0
            Starting point of the first segment (inclusive).
            Must be before the first change point, i.e.,
            ``start < y_sparse["ilocs"].min()``.
        end : optional, default=y_sparse["ilocs].max() + 1
            End point of the last segment (exclusive).
            Must be after the last change point, i.e.,
            ``end > y_sparse["ilocs"].max()``.

        Returns
        -------
        pd.DataFrame with RangeIndex
            Segments corresponding to the change points. Must have the following column:

            * ``"ilocs"`` -  left-closed intervals of iloc based segments.

        Examples
        --------
        >>> import pandas as pd
        >>> from skchange.base import BaseDetector
        >>> change_points = pd.Series([1, 2, 5])
        >>> BaseDetector.change_points_to_segments(change_points, 0, 7)
            ilocs  labels
        0  [0, 1)       0
        1  [1, 2)       1
        2  [2, 5)       2
        3  [5, 7)       3
        """
        if len(y_sparse) == 0:
            return BaseDetector._empty_segments()

        # y_sparse could contain multiple columns, so need to extract the relevant one.
        breaks = y_sparse["ilocs"].values
        if not np.all(np.diff(breaks) > 0):
            raise ValueError("Change points must be sorted in ascending order.")

        # change points can only occur in the range (start, end)
        if start is not None and start >= breaks.min():
            raise ValueError("The start index must be before the first change point.")
        if end is not None and breaks.max() >= end:
            raise ValueError("The end index must be after the last change point.")

        if start is None:
            start = 0
        if end is None:
            end = breaks[-1] + 1

        breaks = np.concatenate(([start], breaks, [end]))
        ilocs = pd.IntervalIndex.from_breaks(breaks, copy=True, closed="left")
        labels = np.arange(len(ilocs), dtype="int64")
        segments = pd.DataFrame({"ilocs": ilocs, "labels": labels})
        return segments

    @staticmethod
    def segments_to_change_points(y_sparse) -> pd.DataFrame:
        """Convert segments to change points.

        Parameters
        ----------
        y_sparse : pd.DataFrame with RangeIndex
            Detected segments. Must have the following column:

            * ``"ilocs"`` -  left-closed intervals of iloc based segments, interpreted
                as the range of indices over which the event takes place.

        Returns
        -------
        pd.DataFrame with RangeIndex
            Corresponding change points for the segments. A change point is defined as
            the left boundary of segments. A left boundary at index 0 is not included.
            Contains the following column:

            * ``"ilocs"`` - the iloc index at which the change points takes place.

        Examples
        --------
        >>> import pandas as pd
        >>> from skchange.base import BaseDetector
        >>> segments = pd.DataFrame({
        ...     "labels": [0, 1, 2],
        ...     "ilocs": pd.IntervalIndex.from_breaks([2, 5, 7, 9], closed="left")
        ... })
        >>> BaseDetector.segments_to_change_points(segments)
         ilocs
        0    2
        1    5
        2    7
        dtype: int64
        """
        if len(y_sparse) == 0:
            return BaseDetector._empty_points()
        segment_starts = y_sparse["ilocs"].array.left
        change_points = segment_starts[segment_starts > 0]
        return pd.DataFrame(change_points, columns=["ilocs"], dtype="int64")

    @staticmethod
    def _empty_points() -> pd.DataFrame:
        """Return an empty sparse series in point format.

        Returns
        -------
        pd.DataFrame
            An empty data frame with an integer ``"ilocs"`` column.
        """
        return pd.DataFrame({"ilocs": pd.Series([], dtype="int64")})

    @staticmethod
    def _empty_segments() -> pd.DataFrame:
        """Return an empty sparse series in segmentation format.

        Returns
        -------
        pd.DataFrame
            An empty data frame with an integer interval ``"ilocs"`` column and an
            integer ``"labels"`` column.
        """
        return pd.DataFrame(
            {
                "ilocs": pd.IntervalIndex([], closed="left"),
                "labels": pd.Series([], dtype="int64"),
            }
        )
