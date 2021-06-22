# -*- coding: utf-8 -*-
"""
Base class template for forecaster scitype.

    class name: BaseAnnotator

Scitype defining methods:
    fitting              - fit(self, X, Y=None, Z=None)
    annotating           - predict(self, X, Y=None, Z=None)

    NOT YET IMPLEMENTED
    updating (temporal)  - update(self, X, Y=None, Z=None)
    update&predict       - update_predict(self, X, X_new, Y=None, Z=None, Z_new=None)
    updating (minibatch) - update_batch(self, X, Y=None, Z=None)

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

import pandas as pd


class BaseAnnotator(BaseEstimator):
    """Base annotator

    The base annotator specifies the methods and method
    signatures that all annotators have to implement.

    Specific implementations of these methods is deferred to concrete
    annotators.
    """

    # default tags
    _tags = {
        "handles-panel": False,  # can handle panel annotations, i.e., list X/y?
        "handles-missing-data": False,  # can handle missing data in X, Y, Z
        "annotation-type": "point",  # can be point, segment or both
        "annotation-labels": "none",  # can be one of, or list-subset of
        #   "label", "outlier", "change"
    }

    def __init__(self):

        self._is_fitted = False

        super(BaseAnnotator, self).__init__()

    def fit(self, X, Y=None, Z=None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame or list of pd.DataFrame
            training data to fit model to, one sequence or multiple sequences
        Y : pd.Series or list of pd.Series, optional
            ground truth annotations for training if annotator is supervised
            feature/label pairs of sequence/annotation are (X[i], Y[i])
        Z : pd.DataFrame, optional
            metadata, rows correspond to index of list in X/Y
        Returns
        -------
        self : returns an instance of self.

        State change
        ------------
        creates fitted model (attributes ending in "_")
        sets _is_fitted flag to true
        """

        if isinstance(X, pd.DataFrame):
            X = [X]

        if isinstance(Y, pd.Series):
            Y = [Y]

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        self._fit(self, X=X, Y=Y, Z=Z)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(self, X, Z=None):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame or list of pd.DataFrame
            data to annotate, one sequence or multiple sequences
        Z : pd.DataFrame, optional; required if passed in fit
            metadata, rows correspond to index of list in X
        Returns
        -------
        Y : pd.Series or list of pd.Series (list iff X is list)
            annotations for sequence(s) in X
            exact format depends on annotation type
        """

        self.check_is_fitted()

        X_was_df = isinstance(X, pd.DataFrame)

        if X_was_df:
            X = [X]

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        Y = self._predict(self, X=X, Z=Z)

        if X_was_df:
            Y = Y[0]

        return Y

    def fit_predict(self, X, X_new=None, Y=None, Z=None, Z_new=None):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame or list of pd.DataFrame
            training data to fit model to, one sequence or multiple sequences
        X_new : pd.DataFrame or list of pd.DataFrame
            data to annotate, one sequence or multiple sequences
            optional, default = X
        Y : pd.Series or list of pd.Series, optional
            ground truth annotations for training if annotator is supervised
            feature/label pairs of sequence/annotation are (X[i], Y[i])
        Z : pd.DataFrame, optional; required if passed in fit
            metadata for X, rows correspond to index of list in X/Y
        Z_new : pd.DataFrame, optional; required if X_new and Z are passed
            metadata for X_new, rows correspond to index of list in X
            if X_new is not passed but Z is, default = Z
        Returns
        -------
        Y_new : pd.Series or list of pd.Series (list iff X is list)
            annotations for sequence(s) in X
            exact format depends on annotation type
        """
        if X_new is None:
            X_new = X
            Z_new = Z

        self.fit(X, Y, Z)

        Y_new = self.predict(X_new, Z_new)

        return Y_new

    def _fit(self, X, Y=None, Z=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : list of pd.DataFrame
            training data to fit model to, one sequence or multiple sequences
        Y : list of pd.Series, optional
            ground truth annotations for training if annotator is supervised
            feature/label pairs of sequence/annotation are (X[i], Y[i])
        Z : pd.DataFrame, optional
            metadata, rows correspond to index of list in X/Y
        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X, Z=None):
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : list of pd.DataFrame
            training data to fit model to, one sequence or multiple sequences
        Z : pd.DataFrame, optional; required if passed in fit
            metadata, rows correspond to index of list in X/Y
        Returns
        -------
        Y : list of pd.Series (list iff X is list)
            annotations for sequence(s) in X
            exact format depends on annotation type
        """
        raise NotImplementedError("abstract method")


# maybe for later:
# class BaseSegmenter(BaseAnnotator)
#
# here you override fit/predict
