# -*- coding: utf-8 -*-
import numpy as np
from sktime.annotation.base._base import BaseSeriesAnnotator

__author__ = ["mloning", "satya-pattnaik", "fkiraly"]

import pandas as pd


class SeriesAnnotatorPyOD(BaseSeriesAnnotator):
    """Transformer that applies outlier detector from pyOD

    Parameters
    ----------
    estimator : PyOD estimator
        See ``https://pyod.readthedocs.io/en/latest/`` documentation for a detailed
        description of all options.
    annotation_format : str - one of "sparse" and "dense"
        "sparse" - sub-series of outliers is returned, with value
        "dense" - series of values is returned with X.index and entry
    annotation_values : str - one of "indicator" and "score"
        "indicator" - value is bool = is this an outlier
        "score" - value is float = outlier score
    """

    _tags = {
        "handles-panel": False,  # can handle panel annotations, i.e., list X/Y?
        "handles-missing-data": False,  # can handle missing data in X, Y
        "annotation-type": "point",  # can be point, segment or both
        "annotation-labels": "outlier",  # can be one of, or list-subset of
        #   "label", "outlier", "change"
    }

    def __init__(
        self, estimator, annotation_format="sparse", annotation_values="indicator"
    ):

        if annotation_format not in ["sparse", "dense"]:
            raise ValueError('annotation_format must be "sparse" or "dense"')

        if annotation_format not in ["indicator", "score"]:
            raise ValueError('annotation_format must be "indicator" or "score"')

        self.estimator = estimator  # pyod estimator
        self.annotation_format = annotation_format
        self.annotation_values = annotation_values

        super(SeriesAnnotatorPyOD, self).__init__()

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

        self.estimator.fit(X_np)

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

        annotation_format = self.annotation_format
        annotation_values = self.annotation_values

        X_np = X.to_numpy()

        Y_np = self.estimator.predict(X_np)

        if annotation_values == "score":
            Y_val_np = self.estimator.decision_function(X_np)
        elif annotation_values == "indicator":
            Y_val_np = Y_np

        if annotation_format == "dense":
            Y = pd.Series(Y_val_np, index=X.index)
        elif annotation_format == "sparse":
            Y_loc = np.where(Y_np)
            Y = pd.Series(Y_val_np[Y_loc], index=X.index[Y_loc])

        return Y
