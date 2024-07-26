#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for annotator base type for time series stream.

    class name: BaseSeriesAnnotator

Scitype defining methods:
    fitting              - fit(self, X, Y=None)
    annotating           - predict(self, X)
    updating (temporal)  - update(self, X, Y=None)
    update&annotate      - update_predict(self, X)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()
"""

__author__ = ["satya-pattnaik ", "fkiraly"]
__all__ = ["BaseSeriesAnnotator"]

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.utils.validation.series import check_series


class BaseSeriesAnnotator(BaseEstimator):
    """Base series annotator.

    Developers should set the task and learning_type tags in the derived class.

    task : str {"segmentation", "change_point_detection", "anomaly_detection"}
        The main annotation task:
        * If ``segmentation``, the annotator divides timeseries into discrete chunks
        based on certain criteria. The same label can be applied at multiple
        disconnected regions of the timeseries.
        * If ``change_point_detection``, the annotator finds points where the
        statistical properties of the timeseries change significantly.
        * If ``anomaly_detection``, the annotator finds points that differ significantly
        from the normal statistical properties of the timeseries.

    learning_type : str {"supervised", "unsupervised"}
        Annotation learning type:
        * If ``supervised``, the annotator learns from labelled data.
        * If ``unsupervised``, the annotator learns from unlabelled data.

    Notes
    -----
    Assumes "predict" data is temporal future of "fit"
    Single time series in both, no meta-data.

    The base series annotator specifies the methods and method
    signatures that all annotators have to implement.

    Specific implementations of these methods is deferred to concrete
    annotators.
    """

    _tags = {
        "object_type": "series-annotator",  # type of object
        "learning_type": "None",  # Tag to determine test in test_all_annotators
        "task": "None",  # Tag to determine test in test_all_annotators
        #
        # todo: distribution_type? we may have to refactor this, seems very soecufuc
        "distribution_type": "None",  # Tag to determine test in test_all_annotators
    }  # for unit test cases

    def __init__(self):
        self.task = self.get_class_tag("task")
        self.learning_type = self.get_class_tag("learning_type")

        self._is_fitted = False

        self._X = None
        self._Y = None

        super().__init__()

    def fit(self, X, Y=None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to (time series).
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Creates fitted model that updates attributes ending in "_". Sets
        _is_fitted flag to True.
        """
        X = check_series(X)

        if Y is not None:
            Y = check_series(Y)

        self._X = X
        self._Y = Y

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        self._fit(X=X, Y=Y)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate (time series).

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X exact format depends on annotation type.
        """
        self.check_is_fitted()

        X = check_series(X)

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        Y = self._predict(X=X)

        return Y

    def transform(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate (time series).

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X. The returned annotations will be in the dense
            format.
        """
        if self.task == "anomaly_detection" or self.task == "change_point_detection":
            Y = self.predict_points(X)
        elif self.task == "segmentation":
            Y = self.predict_segments(X)

        return self.sparse_to_dense(Y, X.index)

    def predict_scores(self, X):
        """Return scores for predicted annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate (time series).

        Returns
        -------
        Y : pd.Series
            Scores for sequence X exact format depends on annotation type.
        """
        self.check_is_fitted()
        X = check_series(X)
        return self._predict_scores(X)

    def update(self, X, Y=None):
        """Update model with new data and optional ground truth annotations.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to update model with (time series).
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        self.check_is_fitted()

        X = check_series(X)

        if Y is not None:
            Y = check_series(Y)

        self._X = X.combine_first(self._X)

        if Y is not None:
            self._Y = Y.combine_first(self._Y)

        self._update(X=X, Y=Y)

        return self

    def update_predict(self, X):
        """Update model with new data and create annotations for it.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to update model with, time series.

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X exact format depends on annotation type.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        X = check_series(X)

        self.update(X=X)
        Y = self.predict(X=X)

        return Y

    def fit_predict(self, X, Y=None):
        """Fit to data, then predict it.

        Fits model to X and Y with given annotation parameters
        and returns the annotations made by the model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        Y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be predicted.

        Returns
        -------
        self : pd.Series
            Annotations for sequence X exact format depends on annotation type.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, Y).predict(X)

    def _fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to time series.
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X):
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate, time series.

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X exact format depends on annotation type.
        """
        raise NotImplementedError("abstract method")

    def _predict_scores(self, X):
        """Return scores for predicted annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate, time series.

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X exact format depends on annotation type.
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, Y=None):
        """Update model with new data and optional ground truth annotations.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Training data to update model with time series
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        # default/fallback: re-fit to all data
        self._fit(self._X, self._Y)

        return self

    def predict_segments(self, X):
        """Predict segments on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate, time series.

        Returns
        -------
        Y : pd.Series
            A series with an index of intervals. Each interval is the range of a
            segment and the corresponding value is the label of the segment.
        """
        if self.task == "anomaly_detection":
            raise RuntimeError(
                "Anomaly detection annotators should not be used for segmentation."
            )
        self.check_is_fitted()
        X = check_series(X)

        if self.task == "change_point_detection":
            return self.change_points_to_segments(
                self.predict_points(X), start=X.index.min(), end=X.index.max()
            )
        elif self.task == "segmentation":
            return self._predict_segments(X)

    def predict_points(self, X):
        """Predict changepoints/anomalies on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate, time series.

        Returns
        -------
        Y : pd.Series
            A series whose values are the changepoints/anomalies in X.
        """
        self.check_is_fitted()
        X = check_series(X)

        if self.task == "anomaly_detection" or self.task == "change_point_detection":
            return self._predict_points(X)
        elif self.task == "segmentation":
            return self.segments_to_change_points(self.predict_segments(X))

    def _predict_segments(self, X):
        """Predict segments on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate, time series.

        Returns
        -------
        Y : pd.Series
            A series with an index of intervals. Each interval is the range of a
            segment and the corresponding value is the label of the segment.
        """
        raise NotImplementedError("abstract method")

    def _predict_points(self, X):
        """Predict changepoints/anomalies on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate, time series.

        Returns
        -------
        Y : pd.Series
            A series whose values are the changepoints/anomalies in X.
        """
        raise NotImplementedError("abstract method")

    @staticmethod
    def sparse_to_dense(y_sparse, index):
        """Convert the sparse output from an annotator to a dense format.

        Parameters
        ----------
        y_sparse : pd.Series
            * If ``y_sparse`` is a series with an index of intervals, it should
              represent segments where each value of the series is label of a segment.
              Unclassified intervals should be labelled -1. Segments must never have
              the label 0.
            * If the index of ``y_sparse`` is not a set of intervals, the values of the
              series should represent the indexes of changepoints/anomalies.
        index : array-like
            Indices that are to be annotated according to ``y_sparse``.

        Returns
        -------
        pd.Series
            A series with an index of ``index`` is returned.
            * If ``y_sparse`` is a series of changepoints/anomalies then the returned
              series is labelled 0 and 1 dependendy on whether the index is associated
              with an anomaly/changepoint. Where 1 means anomaly/changepoint.
            * If ``y_sparse`` is a series of segments then the returned series is
              labelled depending on the segment its indexes fall into. Indexes that
              fall into no segments are labelled -1.

        Examples
        --------
        >>> import pandas as pd
        >>> from sktime.annotation.base._base import BaseSeriesAnnotator
        >>> y_sparse = pd.Series([2, 5, 7])  # Indices of changepoints/anomalies
        >>> index = range(0, 8)
        >>> BaseSeriesAnnotator.sparse_to_dense(y_sparse, index=index)
        0    0
        1    0
        2    1
        3    0
        4    0
        5    1
        6    0
        7    1
        dtype: int64
        >>> y_sparse = pd.Series(
        ...     [1, 2, 1],
        ...     index=pd.IntervalIndex.from_arrays(
        ...         [0, 4, 6], [4, 6, 10], closed="left"
        ...     )
        ... )
        >>> index = range(10)
        >>> BaseSeriesAnnotator.sparse_to_dense(y_sparse, index=index)
        0    1
        1    1
        2    1
        3    1
        4    2
        5    2
        6    1
        7    1
        8    1
        9    1
        dtype: int64
        """
        if isinstance(y_sparse.index.dtype, pd.IntervalDtype):
            # Segmentation case
            y_dense = BaseSeriesAnnotator._sparse_segments_to_dense(y_sparse, index)
            return y_dense
        else:
            # Anomaly/changepoint detection case
            y_dense = BaseSeriesAnnotator._sparse_points_to_dense(y_sparse, index)
            return y_dense

    @staticmethod
    def _sparse_points_to_dense(y_sparse, index):
        """Label the indexes in ``index`` if they are in ``y_sparse``.

        Parameters
        ----------
        y_sparse: pd.Series
            The values of the series must be the indexes of changepoints/anomalies.
        index: array-like
            Array of indexes that are to be labelled according to ``y_sparse``.

        Returns
        -------
        pd.Series
            A series with an index of ``index``. Its values are 1 if the index is in
            y_sparse and 0 otherwise.
        """
        y_dense = pd.Series(np.zeros(len(index)), index=index, dtype="int64")
        y_dense[y_sparse.values] = 1
        return y_dense

    @staticmethod
    def _sparse_segments_to_dense(y_sparse, index):
        """Find the label for each index in ``index`` from sparse segments.

        Parameters
        ----------
        y_sparse : pd.Series
            A sparse representation of segments. The index must be the pandas interval
            datatype and the values must be the integer labels of the segments.
        index : array-like
            List of indexes that are to be labelled according to ``y_sparse``.

        Returns
        -------
        pd.Series
            A series with the same index as ``index`` where each index is labelled
            according to ``y_sparse``. Indexes that do not fall within any index are
            labelled -1.
        """
        if y_sparse.index.is_overlapping:
            raise NotImplementedError(
                "Cannot convert overlapping segments to a dense format yet."
            )

        interval_indexes = y_sparse.index.get_indexer(index)

        # Negative indexes do not fall within any interval so they are ignored
        interval_labels = y_sparse.iloc[
            interval_indexes[interval_indexes >= 0]
        ].to_numpy()

        # -1 is used to represent points do not fall within a segment
        labels_dense = interval_indexes.copy()
        labels_dense[labels_dense >= 0] = interval_labels

        y_dense = pd.Series(labels_dense, index=index)
        return y_dense

    @staticmethod
    def dense_to_sparse(y_dense):
        """Convert the dense output from an annotator to a sparse format.

        Parameters
        ----------
        y_dense : pd.Series
            * If ``y_sparse`` contains only 1's and 0's, the 1's represent change
              points or anomalies.
            * If ``y_sparse`` contains only contains integers greater than 0, it is an
              an array of segments.

        Returns
        -------
        pd.Series
            * If ``y_sparse`` is a series of changepoints/anomalies, a pandas series
              will be returned containing the indexes of the changepoints/anomalies
            * If ``y_sparse`` is a series of segments, a series with an interval
              datatype index will be returned. The values of the series will be the
              labels of segments.
        """
        if 0 in y_dense.values:
            # y_dense is a series of change points
            change_points = np.where(y_dense.values != 0)[0]
            return pd.Series(change_points)
        else:
            segment_start_indexes = np.where(y_dense.diff() != 0)[0]
            segment_end_indexes = np.roll(segment_start_indexes, -1)

            # The final index is always the end of a segment
            segment_end_indexes[-1] = y_dense.index[-1]

            segment_labels = y_dense.iloc[segment_start_indexes].to_numpy()
            interval_index = pd.IntervalIndex.from_arrays(
                segment_start_indexes, segment_end_indexes, closed="left"
            )
            y_sparse = pd.Series(segment_labels, index=interval_index)

            # -1 represents unclassified regions so we remove them
            y_sparse = y_sparse.loc[y_sparse != -1]
            return y_sparse

    @staticmethod
    def change_points_to_segments(y_sparse, start=None, end=None):
        """Convert an series of change point indexes to segments.

        Parameters
        ----------
        y_sparse : pd.Series
            A series containing the indexes of change points.
        start : optional
            Starting point of the first segment.
        end : optional
            Ending point of the last segment

        Returns
        -------
        pd.Series
            A series with an interval index indicating the start and end points of the
            segments. The values of the series are the labels of the segments.

        Examples
        --------
        >>> import pandas as pd
        >>> from sktime.annotation.base._base import BaseSeriesAnnotator
        >>> change_points = pd.Series([1, 2, 5])
        >>> BaseSeriesAnnotator.change_points_to_segments(change_points, 0, 7)
        [0, 1)   -1
        [1, 2)    1
        [2, 5)    2
        [5, 7)    3
        dtype: int64
        """
        breaks = y_sparse.values

        if start > breaks.min():
            raise ValueError(
                "The starting index must be before the first change point."
            )
        first_change_point = breaks.min()

        if start is not None:
            breaks = np.insert(breaks, 0, start)
        if end is not None:
            breaks = np.append(breaks, end)

        index = pd.IntervalIndex.from_breaks(breaks, copy=True, closed="left")
        segments = pd.Series(0, index=index)

        in_range = index.left >= first_change_point

        number_of_segments = in_range.sum()
        segments.loc[in_range] = range(1, number_of_segments + 1)
        segments.loc[~in_range] = -1

        return segments

    @staticmethod
    def segments_to_change_points(y_sparse):
        """Convert segments to change points.

        Parameters
        ----------
        y_sparse : pd.DataFrame
            A series of segments. The index must be the interval data type and the
            values should be the integer labels of the segments.

        Returns
        -------
        pd.Series
            A series containing the indexes of the start of each segment.

        Examples
        --------
        >>> import pandas as pd
        >>> from sktime.annotation.base._base import BaseSeriesAnnotator
        >>> segments = pd.Series(
        ...     [3, -1, 2],
        ...     index=pd.IntervalIndex.from_breaks([2, 5, 7, 9], closed="left")
        ... )
        >>> BaseSeriesAnnotator.segments_to_change_points(segments)
        0    2
        1    5
        2    7
        dtype: int64
        """
        change_points = pd.Series(y_sparse.index.left)
        return change_points
