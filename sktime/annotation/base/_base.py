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

from sktime.base import BaseEstimator
from sktime.utils.validation.annotation import check_learning_type, check_task
from sktime.utils.validation.series import check_series


class BaseSeriesAnnotator(BaseEstimator):
    """Base series annotator.

    Developers should set the task and learning_type tags in the derived class.

    task : str {"segmentation", "change_point_detection", "anomaly_detection"}
        The main annotation task:
        * If `segmentation`, the annotator divides timeseries into discrete chunks
        based on certain criteria. The same label can be applied at mulitple
        disconnected regions of the timeseries.
        * If `change_point_detection`, the annotator finds points where the statistical
        properties of the timeseries change significantly.
        * If `anomaly_detection`, the annotator finds points that differ significantly
        from the normal statistical properties of the timeseries.

    learning_type : str {"supervised", "unsupervised"}
        Annotation learning type:
        * If `supervised`, the annotator learns from labelled data.
        * If `unsupervised`, the annotator learns from unlabelled data.

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
        check_learning_type(self.learning_type)
        check_task(self.task)
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
        self.check_is_fitted()

        X = check_series(X)

        Y = self.predict(X=X)

        return self.sparse_to_dense(Y)

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

    @staticmethod
    def sparse_to_dense(y_sparse):
        """Convert the sparse output from an annotator to a dense format.

        Parameters
        ----------
        y_sparse : np.ndarray
            If `y_sparse` is a 1D array then it should contain the index locations of
            changepoints/anomalies.

        Returns
        -------
        np.ndarray
            If `y_sparse` is a 1D array of changepoint/anomaly indices then a 1D array
            of 0's and 1's is returned. The array is 1 at the indices of the
            anomalies/changepoints.

        Examples
        --------
        >>> import numpy as np
        >>> from sktime.annotation.base._base import BaseSeriesAnnotator
        >>> y_sparse = np.array([2, 5, 7])  # Indices of changepoints/anomalies
        >>> BaseSeriesAnnotator.sparse_to_dense(y_sparse)
        array([0, 0, 1, 0, 0, 1, 0, 1], dtype=int32)

        TODO: Handle the 2D case for segmentation.
        """
        if y_sparse.ndim == 1:
            y_dense = np.zeros(y_sparse[-1] + 1, dtype=np.int32)
            np.put(y_dense, y_sparse, 1)
            return y_dense
        else:
            raise NotImplementedError("Cannot handle the 2D case yet.")

    @staticmethod
    def dense_to_sparse(y_dense):
        """Convert the dense output from an annotator to a dense format.

        Parameters
        ----------
        y_dense : np.ndarray
            The array must be 1D.
            * If `y_sparse` contains only 1's and 0's the 1's represent change points
              or anomalies
            * If `y_sparse` contains only contains integers greater than 0, it is an
              an array of segments.

        Returns
        -------
        np.ndarray
            * If `y_sparse` is an array of changepoints/anomalies, the returned array
              will be 1D and contains the indexes of the the changepoints/anomalies
            * If `y_sparse` is an array of segments, a 2D array is returned. The first
              column contains the labels of the segments, the second column contains
              the starting points of the segments.

        Examples
        --------
        >>> import numpy as np
        >>> from sktime.annotation.base._base import BaseSeriesAnnotator
        >>> change_points = np.array([1, 0, 0, 1, 1, 0, 1])
        >>> BaseSeriesAnnotator.dense_to_sparse(change_points)
        array([1, 2, 5])
        >>> segments = np.array([1, 2, 2, 3, 3, 2])
        >>> BaseSeriesAnnotator.dense_to_sparse(segments)
        array([[1, 0],
               [2, 1],
               [3, 3],
               [2, 5]])
        """
        if y_dense.min() == 0:
            return np.where(y_dense == 0)[0]
        elif y_dense.min() == 1:
            # Prepend zero so the first point is always the start of a segment
            diff = np.diff(y_dense, prepend=0)
            segment_start_indexes = np.where(diff != 0)[0]
            segment_labels = y_dense[diff.astype(bool)]
            return np.stack([segment_labels, segment_start_indexes]).T

    @staticmethod
    def change_points_to_segments(y_sparse):
        """Convert an array of change point indexes to segments.

        Parameters
        ----------
        y_sparse : np.ndarray
            A 1D array containing the indexes of change points.

        Returns
        -------
        np.ndarray
            A 2D array where the first column columns that label of segements and the
            second column contains the starting indexes of the change points.

        Examples
        --------
        >>> import numpy as np
        >>> from sktime.annotation.base._base import BaseSeriesAnnotator
        >>> change_points = np.array([1, 2, 5])
        >>> BaseSeriesAnnotator.change_points_to_segments(change_points)
        array([[1, 0],
               [2, 1],
               [3, 2],
               [4, 5]])
        """
        if y_sparse[0] != 0:
            # Insert a 0 at the start so the points before the first anomaly are
            # considered a segment
            y_sparse = np.insert(y_sparse, 0, 0)
        labels = np.arange(1, len(y_sparse) + 1)
        return np.stack([labels, y_sparse]).T

    @staticmethod
    def segments_to_change_points(y_sparse):
        """Convert 2D array of segments to a 1D array of change points.

        Parameters
        ----------
        y_sparse : np.ndarray
            A 2D array of segments. The first column contains the labels of the
            segments and the second column contains the indexes of the starting points
            of the segments.

        Returns
        -------
        np.ndarray
            A 1D array containing the indexes of the change points in `y_sparse`.
        """
        # The first segment should start at index 0 which is not a change point so we
        # ignore it
        return y_sparse[1:, 1]
