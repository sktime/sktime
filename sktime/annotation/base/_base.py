#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)


from sktime.base import BaseEstimator


class BaseAnnotator(BaseEstimator):
    """Base annotator

    The base annotator specifies the methods and method
    signatures that all annotators have to implement.

    Specific implementations of these methods is deferred to concrete
    annotators.
    """

    def __init__(self):
        self._is_fitted = False
        super(BaseAnnotator, self).__init__()

    def fit(self, Z, X=None):
        """Fit to training data.

        Parameters
        ----------
        Z : pd.Series
            Target time series to which to fit the annotator.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

    # Franz suggestion for fit
    def fit(self, X, Y=None, Z=None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame -> internally, convert to 1-element-list
            list of pd.DataFrame (e.g., look at hmmlearn or seglearn)
            series that are annotated, for training
        Y : pd.Series?DataFrame or list thereof -optional
            ground truth annotations if annotator is *supervised*
        Z : pd.DataFrame
            rows correspond to index of list in X - metadata
        Returns
        -------
        self : returns an instance of self.
        """

    # Franz suggestion for predict
    def predict(self, X, Z=None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame -> internally, convert to 1-element-list
            list of pd.DataFrame (e.g., look at hmmlearn or seglearn)
            series that are annotated, for training
        Z : pd.DataFrame - only passed if passed in fit
            rows correspond to index of list in X - metadata
        Returns
        -------
        Y : pd.DataFrame?Series or list thereof
            ground truth annotations if annotator is *supervised*
                -> exact format depends on annotation type - to be specified
        """

        check
        check
        mroe check

        return _predict(X=X, Z=Z)


example Y:

pd.DataFrame({'event_time' : [ 10, 100, 500], 'event_type' : ['outlier', 'changepoint', 'hand movement']})

#for each annotation type, we could think about having *two* supported formats - sparse and dense
# sparse = only annotation times are indexed; dense = all times in X.index are indexed


    def predict(self, Z, X=None):
        """Returns a transformed version of Z.

        Parameters
        ----------
        Z : pd.Series
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        z : pd.Series
            Transformed(by annotator) time series.
        """
        raise NotImplementedError("abstract method")



class BaseSegmenter(BaseAnnotator)


here you override fit/predict