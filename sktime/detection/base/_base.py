#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for detector base type for time series streams.

    class name: BaseDetector

Scitype defining methods:
    fitting              - fit(self, X, y=None)
    annotating           - predict(self, X)
    updating (temporal)  - update(self, X, y=None)
    update&annotate      - update_predict(self, X, y=None)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()
"""

# todo 0.37.0: remove BaseSeriesAnnotator
__author__ = ["fkiraly", "tveten", "alex-jg3", "satya-pattnaik"]
__all__ = ["BaseDetector", "BaseSeriesAnnotator"]

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.utils.validation.series import check_series
from sktime.utils.warnings import warn


class BaseDetector(BaseEstimator):
    """Base class for time series detectors.

    Developers should set the task and learning_type tags in the derived class.

    task : str {"segmentation", "change_point_detection", "anomaly_detection"}
        The main detection task:

        * If ``segmentation``, the detector divides timeseries into discrete chunks
        based on certain criteria. The same label can be applied at multiple
        disconnected regions of the timeseries.
        * If ``change_point_detection``, the detector finds points where the
        statistical properties of the timeseries change significantly.
        * If ``anomaly_detection``, the detector finds points that differ significantly
        from the normal statistical properties of the timeseries.

    learning_type : str {"supervised", "unsupervised"}
        Detection learning type:

        * If ``supervised``, the detector learns from labelled data.
        * If ``unsupervised``, the detector learns from unlabelled data.

    Notes
    -----
    Assumes "predict" data is temporal future of "fit"
    Single time series in both, no meta-data.

    The base series detector specifies the methods and method
    signatures that all detectors have to implement.

    Specific implementations of these methods is deferred to concrete detectors.
    """

    _tags = {
        # todo 0.37.0 switch order of series-annotator and detector
        # todo 1.0.0 - remove series-annotator
        "object_type": ["series-annotator", "detector"],  # type of object
        "learning_type": "None",  # Tag to determine test in test_all_annotators
        "task": "None",  # Tag to determine test in test_all_annotators
        #
        # todo: distribution_type? we may have to refactor this, seems very soecufuc
        "distribution_type": "None",  # Tag to determine test in test_all_annotators
        "X_inner_mtype": "pd.DataFrame",  # Tag to determine test in test_all_annotators
    }

    def __init__(self):
        self._is_fitted = False

        self._X = None
        self._Y = None

        task = self.get_tag("task")
        learning_type = self.get_tag("learning_type")

        super().__init__()

        self.set_tags(**{"task": task, "learning_type": learning_type})

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated AnnotatorPipeline.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        AnnotatorPipeline object,
            concatenation of ``other`` (first) with ``self`` (last).
            not nested, contains only non-AnnotatorPipeline ``sktime`` steps
        """
        from sktime.annotation.compose import AnnotatorPipeline
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor
        from sktime.utils.sklearn import is_sklearn_transformer

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformedTargetForecaster does the rest, e.g., dispatch on other
        if isinstance(other, BaseTransformer):
            self_as_pipeline = AnnotatorPipeline(steps=[self])
            return other * self_as_pipeline
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    # todo 0.37.0: remove the Y parameter and related handling
    def fit(self, X, y=None, Y=None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Training data to fit model to (time series).
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Creates fitted model that updates attributes ending in "_". Sets
        _is_fitted flag to True.
        """
        X_inner = self._check_X(X)

        if Y is not None:
            warn(
                "Warning: the Y parameter in detection/annotation algorithms "
                "is deprecated and will be removed in the 0.37.0 release. "
                "Users should use the y parameter instead. "
                "Until the 0.37.0 release, the Y parameter will be used if "
                "no y parameter is provided, ensuring backwards compatibility.",
                stacklevel=2,
            )

        if Y is not None and y is None:
            y = Y

        self._X = X
        self._y = y

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        if _method_has_arg(self._fit, "y"):
            self._fit(X=X, y=y)
        elif _method_has_arg(self._fit, "Y"):
            self._fit(X=X, Y=y)
            warn(
                "Warning: the Y parameter in detection/annotation algorithms "
                "is deprecated and will be removed in the 0.37.0 release. "
                "Users should use the y parameter instead. "
                f"The class {self.__class__.__name__} uses the Y parameter "
                "internally in _fit, this should be replaced with y by a maintainer. "
                f"Until the 0.37.0 release, this will raise no exceptions, "
                "ensuring backwards compatibility.",
                stacklevel=2,
            )
        else:
            self._fit(X=X_inner)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(self, X):
        """Create labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the values are integer indices of the changepoints/anomalies.
            * If ``task`` is "segmentation", the values are ``pd.Interval`` objects.
        """
        self.check_is_fitted()

        X_inner = self._check_X(X)

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        y = self._predict(X=X_inner)

        return y

    def transform(self, X):
        """Create labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with same index as X
            Labels for sequence ``X``.

            * If ``task`` is ``"anomaly_detection"``, the values are integer labels.
              A value of 0 indicatesthat ``X``, at the same time index, has no anomaly.
              Other values indicate an anomaly.
              Most detectors will return 0 or 1, but some may return more values,
              if they can detect different types of anomalies.
              indicating whether ``X``, at the same
              index, is an anomaly, 0 for no, 1 for yes.
            * If ``task`` is ``"changepoint_detection"``, the values are integer labels,
              indicating labels for segments between changepoints.
              Possible labels are integers starting from 0.
            * If ``task`` is "segmentation", the values are integer labels of the
              segments. Possible labels are integers starting from 0.
        """
        y = self.predict(X)
        return self.sparse_to_dense(y, X.index)

    def transform_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to label (time series).

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Scores for sequence ``X``.
        """
        self.check_is_fitted()

        X_inner = self._check_X(X)

        return self._transform_scores(X_inner)

    def predict_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to label (time series).

        Returns
        -------
        scores : pd.DataFrame with same index as return of predict
            Scores for prediction of sequence ``X``.
        """
        self.check_is_fitted()

        X_inner = self._check_X(X)

        return self._predict_scores(X_inner)

    def update(self, X, y=None, Y=None):
        """Update model with new data and optional ground truth labels.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Training data to update model with (time series).
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        self.check_is_fitted()

        X_inner = self._check_X(X)

        if Y is not None:
            warn(
                "Warning: the Y parameter in detection/annotation algorithms "
                "is deprecated and will be removed in the 0.37.0 release. "
                "Users should use the y parameter instead. "
                "Until the 0.37.0 release, the Y parameter will be used if "
                "no y parameter is provided, ensuring backwards compatibility.",
                stacklevel=2,
            )

        if y is None and Y is not None:
            y = Y

        self._X = X_inner.combine_first(self._X)

        if y is not None:
            self._y = y.combine_first(self._y)

        if _method_has_arg(self._update, "y"):
            self._update(X=X_inner, y=y)
        elif _method_has_arg(self._update, "Y"):
            self._update(X=X_inner, Y=y)
        else:
            self._update(X=X_inner)

        return self

    def update_predict(self, X, y=None):
        """Update model with new data and create labels for it.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Training data to update model with, time series.
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        Y : pd.Series
            Labels for sequence X exact format depends on detection type.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        X_inner = self._check_X(X)

        self.update(X=X, y=y)
        y = self.predict(X=X_inner)

        return y

    # todo 0.37.0: remove Y argument
    def fit_predict(self, X, y=None, Y=None):
        """Fit to data, then predict it.

        Fits model to X and Y with given detection parameters
        and returns the detection labels produced by the model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be predicted.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the values are integer indices of the changepoints/anomalies.
            * If ``task`` is "segmentation", the values are ``pd.Interval`` objects.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y=y, Y=Y).predict(X)

    # todo 0.37.0: remove Y argument
    def fit_transform(self, X, y=None, Y=None):
        """Fit to data, then transform it.

        Fits model to X and Y with given detection parameters
        and returns the detection labels made by the model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        Y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be predicted.

        Returns
        -------
        y : pd.DataFrame with same index as X
            Labels for sequence ``X``.

            * If ``task`` is ``"anomaly_detection"``, the values are integer labels.
              A value of 0 indicatesthat ``X``, at the same time index, has no anomaly.
              Other values indicate an anomaly.
              Most detectors will return 0 or 1, but some may return more values,
              if they can detect different types of anomalies.
              indicating whether ``X``, at the same
              index, is an anomaly, 0 for no, 1 for yes.
            * If ``task`` is ``"changepoint_detection"``, the values are integer labels,
              indicating labels for segments between changepoints.
              Possible labels are integers starting from 0.
            * If ``task`` is "segmentation", the values are integer labels of the
              segments. Possible labels are integers starting from 0.
        """
        y = self.fit_predict(X, y=y, Y=Y)
        return self.sparse_to_dense(y, index=X.index)

    def _check_X(self, X):
        """Check input data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed

        Returns
        -------
        X : X_inner_mtype
            Data to be transformed
        """
        return X
        # this causes errors, we need to investigate
        # X_inner_mtype = self.get_tag("X_inner_mtype")
        # X_inner = convert_to(X, X_inner_mtype)
        # return X_inner

    def _fit(self, X, y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to time series.
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

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
        """Create labels on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            Labels for sequence X exact format depends on detection type.
        """
        raise NotImplementedError("abstract method")

    def _predict_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            Labels for sequence X exact format depends on detection type.
        """
        raise NotImplementedError("abstract method")

    def _transform_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Scores for sequence ``X``.
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, y=None):
        """Update model with new data and optional ground truth labels.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Training data to update model with time series
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        # default/fallback: re-fit to all data
        self._fit(self._X, self._y)

        return self

    def predict_segments(self, X):
        """Predict segments on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.Series with IntervalIndex
            A series with an index of intervals. Each interval is the range of a
            segment and the corresponding value is the label of the segment.

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the intervals are intervals between changepoints/anomalies, and
              the labels are consecutive integers starting from 0.
            * If ``task`` is "segmentation", the values are segmentation labels.
        """
        self.check_is_fitted()
        X = check_series(X)

        task = self.get_tag("task")
        if task in ["anomaly_detection", "change_point_detection"]:
            return self.change_points_to_segments(
                self.predict_points(X), start=X.index.min(), end=X.index.max()
            )
        elif task == "segmentation":
            return self._predict_segments(X)

    def predict_points(self, X):
        """Predict changepoints/anomalies on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the values are integer indices of the changepoints/anomalies.
            * If ``task`` is "segmentation", the values are consecutive
              segment boundaries.
        """
        self.check_is_fitted()
        X = check_series(X)

        task = self.get_tag("task")
        if task in ["anomaly_detection", "change_point_detection"]:
            return self._predict_points(X)
        elif task == "segmentation":
            return self.segments_to_change_points(self.predict_segments(X))

    def _predict_segments(self, X):
        """Predict segments on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            A series with an index of intervals. Each interval is the range of a
            segment and the corresponding value is the label of the segment.
        """
        return self._predict(X)

    def _predict_points(self, X):
        """Predict changepoints/anomalies on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            A series whose values are the changepoints/anomalies in X.
        """
        return self._predict(X)

    @staticmethod
    def sparse_to_dense(y_sparse, index):
        """Convert the sparse output from an detector to a dense format.

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
        >>> from sktime.detection.base import BaseDetector
        >>> y_sparse = pd.Series([2, 5, 7])  # Indices of changepoints/anomalies
        >>> index = range(0, 8)
        >>> BaseDetector.sparse_to_dense(y_sparse, index=index)
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
        >>> BaseDetector.sparse_to_dense(y_sparse, index=index)
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
            y_dense = BaseDetector._sparse_segments_to_dense(y_sparse, index)
            return y_dense
        else:
            # Anomaly/changepoint detection case
            y_dense = BaseDetector._sparse_points_to_dense(y_sparse, index)
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
        """Convert the dense output from an detector to a sparse format.

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
        >>> from sktime.detection.base import BaseDetector
        >>> change_points = pd.Series([1, 2, 5])
        >>> BaseDetector.change_points_to_segments(change_points, 0, 7)
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
        >>> from sktime.detection.base import BaseDetector
        >>> segments = pd.Series(
        ...     [3, -1, 2],
        ...     index=pd.IntervalIndex.from_breaks([2, 5, 7, 9], closed="left")
        ... )
        >>> BaseDetector.segments_to_change_points(segments)
        0    2
        1    5
        2    7
        dtype: int64
        """
        change_points = pd.Series(y_sparse.index.left)
        return change_points


class BaseSeriesAnnotator(BaseDetector):
    """Base class for time series detectors - DEPRECATED - use BaseDetector instead."""

    def __init__(self):
        super().__init__()
        warn(
            "Warning: BaseSeriesAnnotator is deprecated. "
            "Extension developers should use BaseDetector instead, "
            "from sktime.detection.base, this is a replacement with "
            "equivalent functionality. "
            "The BaseSeriesAnnotator will be removed in the 0.37.0 release.",
            stacklevel=2,
        )


# todo 0.37.0: remove this
def _method_has_arg(method, arg="y"):
    """Return if transformer.method has a parameter, and whether it has a default.

    Parameters
    ----------
    method : callable
        method to check
    arg : str, optional, default="y"
        parameter name to check

    Returns
    -------
    has_param : bool
        whether the method ``method`` has a parameter with name ``arg``
    """
    from inspect import signature

    method_params = list(signature(method).parameters.keys())
    return arg in method_params
