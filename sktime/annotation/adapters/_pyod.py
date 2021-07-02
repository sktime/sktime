# -*- coding: utf-8 -*-
import numpy as np
from sktime.annotation.base._base import BaseSeriesAnnotator

__author__ = ["mloning", "satya-pattnaik", "fkiraly"]

import pandas as pd

from sklearn import clone


class PyODAnnotator(BaseSeriesAnnotator):
    """Transformer that applies outlier detector from pyOD

    Parameters
    ----------
    estimator : PyOD estimator
        See ``https://pyod.readthedocs.io/en/latest/`` documentation for a detailed
        description of all options.
    fmt : str {"dense", "sparse"}, optional (default="dense")
        Annotation output format:
        * If "sparse", a sub-series of labels for only the outliers in X is returned,
        * If "dense", a series of labels for all values in X is returned.
    labels : str {"indicator", "score"}, optional (default="indicator")
        Annotation output labels:
        * If "indicator", returned values are boolean, indicating whether a value is an
        outlier,
        * If "score", returned values are floats, giving the outlier score.
    """

    def __init__(self, estimator, fmt="dense", labels="indicator"):
        self.estimator = estimator  # pyod estimator
        super(PyODAnnotator, self).__init__(fmt=fmt, labels=labels)

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

        X_np = X.to_numpy()

        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_np)

        return self

    def _predict(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """

        fmt = self.fmt
        labels = self.labels

        X_np = X.to_numpy()

        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)

        Y_np = self.estimator_.predict(X_np)

        if labels == "score":
            Y_val_np = self.estimator_.decision_function(X_np)
        elif labels == "indicator":
            Y_val_np = Y_np

        if fmt == "dense":
            Y = pd.Series(Y_val_np, index=X.index)
        elif fmt == "sparse":
            Y_loc = np.where(Y_np)
            Y = pd.Series(Y_val_np[Y_loc], index=X.index[Y_loc])

        return Y
