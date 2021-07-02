# -*- coding: utf-8 -*-
"""
Base class template for annotator base type for time series stream

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

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["satya-pattnaik ", "fkiraly"]

from sktime.base import BaseEstimator
from sktime.utils.validation.annotation import check_fmt
from sktime.utils.validation.annotation import check_labels

from sktime.utils.validation.series import check_series


class BaseSeriesAnnotator(BaseEstimator):
    """Base series annotator.

    Parameters
    ----------
    fmt : str {"dense", "sparse"}, optional (default="dense")
        Annotation output format:
        * If "sparse", a sub-series of labels for only the outliers in X is returned,
        * If "dense", a series of labels for all values in X is returned.
    labels : str {"indicator", "score"}, optional (default="indicator")
        Annotation output labels:
        * If "indicator", returned values are boolean, indicating whether a value is an
        outlier,
        * If "score", returned values are floats, giving the outlier score.

    Notes
    -----
    Assumes "predict" data is temporal future of "fit"
        single time series in both, no meta-data

    The base series annotator specifies the methods and method
    signatures that all annotators have to implement.

    Specific implementations of these methods is deferred to concrete
    annotators.
    """

    def __init__(self, fmt="dense", labels="indicator"):
        self.fmt = fmt
        self.labels = labels

        self._is_fitted = False

        self._X = None
        self._Y = None

        super(BaseSeriesAnnotator, self).__init__()

    def fit(self, X, Y=None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised
        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        sets _is_fitted flag to true
        """
        check_labels(self.labels)
        check_fmt(self.fmt)
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
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """

        self.check_is_fitted()

        X = check_series(X)

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        Y = self._predict(X=X)

        return Y

    def update(self, X, Y=None):
        """update model with new data and optional ground truth annotations

        Parameters
        ----------
        X : pd.DataFrame
            training data to update model with, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised
        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        updates fitted model (attributes ending in "_")
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
        """update model with new data and create annotations for it

        Parameters
        ----------
        X : pd.DataFrame
            training data to update model with, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type

        State change
        ------------
        updates fitted model (attributes ending in "_")
        """

        X = check_series(X)

        self.update(X=X)
        Y = self.predict(X=X)

        return Y

    def _fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised
        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X):
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, Y=None):
        """update model with new data and optional ground truth annotations

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to update model with, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised
        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        updates fitted model (attributes ending in "_")
        """

        # default/fallback: re-fit to all data
        self._fit(self._X, self._Y)

        return self
