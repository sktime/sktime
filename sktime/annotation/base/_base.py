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

from sktime.utils.validation.series import check_series


# decorator to alias "X" argument as "Z" for transformer use
def alias_X_as_Z(inner):

    def outer(X=None, Z=None, **kwargs):

        if X is None and Z is None:
            raise ValueError("exactly one of X and Z must be passed, found none")

        if X is not None and Z is not None:
            raise ValueError("exactly one of X and Z must be passed, found both")

        if X is None:
            X = Z

        return inner(X, **kwargs)

    return outer


class BaseSeriesAnnotator(BaseEstimator):
    """Base stream annotator
    assumes "predict" data is temporal future of "fit"
        single time series in both, no meta-data

    The base stream annotator specifies the methods and method
    signatures that all annotators have to implement.

    Specific implementations of these methods is deferred to concrete
    annotators.

    # default tags
    _tags = {
        "handles-panel": False,  # can handle panel annotations, i.e., list X/Y?
        "handles-missing-data": False,  # can handle missing data in X, Y
        "annotation-type": "point",  # can be point, segment or both
        "annotation-labels": "none",  # can be one of, or list-subset of
        #   "label", "outlier", "change"
    }
    """

    def __init__(self):

        self._is_fitted = False

        self._X = None
        self._Y = None

        super(BaseSeriesAnnotator, self).__init__()

    @alias_X_as_Z
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

    @alias_X_as_Z
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

        self._X = self._X.append(X)

        if Y is not None:
            self._Y.append(Y)

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

    def transform(self, Z):
        """transform series - interface alias for predict, for use as transformer

        Parameters
        ----------
        Z : pd.DataFrame
            training data to fit model to, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence Z
            exact format depends on annotation type
        """

        return self.predict(X=Z)

    def update_transform(self, Z):
        """update model with new data and create annotations for it

        Parameters
        ----------
        Z : pd.DataFrame
            training data to update model with, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence Z
            exact format depends on annotation type

        State change
        ------------
        updates fitted model (attributes ending in "_")
        """

        Z = check_series(Z)

        self.update(Z=Z)
        Y = self.transform(Z=Z)

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
