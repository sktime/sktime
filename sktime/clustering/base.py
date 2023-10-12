"""Base class for clustering."""
__author__ = ["chrisholder", "TonyBagnall", "achieveordie"]
__all__ = ["BaseClusterer"]

import time
from typing import Any, Union

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.datatypes import check_is_scitype, convert_to, scitype_to_mtype
from sktime.utils.sklearn import is_sklearn_transformer
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation._dependencies import _check_estimator_deps

# Valid input types for clustering
TimeSeriesInstances = Union[pd.DataFrame, np.ndarray]


class BaseClusterer(BaseEstimator):
    """Abstract base class for time series clusterer.

    Parameters
    ----------
    n_clusters: int, defaults = None
        Number of clusters for model.
    """

    _tags = {
        "object_type": "clusterer",  # type of object
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict accept, usually
        # this is either "numpy3D" or "nested_univ" (nested pd.DataFrame). Other
        # types are allowable, see datatypes/panel/_registry.py for options.
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:multithreading": False,
    }

    def __init__(self, n_clusters: int = None):
        self.fit_time_ = 0
        self._class_dictionary = {}
        self._threads_to_use = 1
        self.n_clusters = n_clusters
        super().__init__()
        _check_estimator_deps(self)

    def __rmul__(self, other):
        """Magic * method, return concatenated ClustererPipeline, transformers on left.

        Overloaded multiplication operation for clusterers. Implemented for `other`
        being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ClustererPipeline object, concatenation of `other` (first) with `self` (last).
        """
        from sktime.clustering.compose import ClustererPipeline
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor

        # behaviour is implemented only if other inherits from BaseTransformer
        #  in that case, distinctions arise from whether self or other is a pipeline
        #  todo: this can probably be simplified further with "zero length" pipelines
        if isinstance(other, BaseTransformer):
            # ClustererPipeline already has the dunder method defined
            if isinstance(self, ClustererPipeline):
                return other * self
            # if other is a TransformerPipeline but self is not, first unwrap it
            elif isinstance(other, TransformerPipeline):
                return ClustererPipeline(clusterer=self, transformers=other.steps)
            # if neither self nor other are a pipeline, construct a ClustererPipeline
            else:
                return ClustererPipeline(clusterer=self, transformers=[other])
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    def fit(self, X: TimeSeriesInstances, y=None) -> BaseEstimator:
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : Training time series instances to cluster. np.ndarray (2d or 3d array of
        shape (n_instances, series_length) or shape (n_instances, n_dimensions,
        series_length)) or pd.DataFrame (where each column is a dimension, each cell
        is a pd.Series (any number of dimensions, equal or unequal length series)).
        Converted to type _tags["X_inner_mtype"]
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        # reset estimator at the start of fit
        self.reset()

        X = self._check_clusterer_input(X)

        multithread = self.get_tag("capability:multithreading")
        if multithread:
            try:
                self._threads_to_use = check_n_jobs(self.n_jobs)
            except NameError:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )

        start = int(round(time.time() * 1000))
        self._fit(X)
        self.fit_time_ = int(round(time.time() * 1000)) - start
        self._is_fitted = True
        return self

    def predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length)) or pd.DataFrame (where each
            column is a dimension, each cell is a pd.Series (any number of dimensions,
            equal or unequal length series)).
            Time series instances to predict their cluster indexes.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        self.check_is_fitted()
        X = self._check_clusterer_input(X)
        return self._predict(X)

    def fit_predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each time series.

        Convenience method; equivalent of calling fit(X) followed by predict(X)

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length)) or pd.DataFrame (where each
            column is a dimension, each cell is a pd.Series (any number of dimensions,
            equal or unequal length series)).
            Time series instances to train clusterer and then have indexes each belong
            to return.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        self.fit(X)
        return self.predict(X)

    def predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb


        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        self.check_is_fitted()
        X = self._check_clusterer_input(X)
        return self._predict_proba(X)

    def score(self, X, y=None) -> float:
        """Score the quality of the clusterer.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length)) or pd.DataFrame (where each
            column is a dimension, each cell is a pd.Series (any number of dimensions,
            equal or unequal length series)).
            Time series instances to train clusterer and then have indexes each belong
            to return.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        score : float
            Score of the clusterer.
        """
        self.check_is_fitted()
        X = self._check_clusterer_input(X)
        return self._score(X, y)

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        preds = self._predict(X)
        n_instances = len(preds)
        if hasattr(self, "n_clusters") and self.n_clusters is not None:
            n_clusters = self.n_clusters
        else:
            n_clusters = max(preds) + 1
        dists = np.zeros((X.shape[0], n_clusters))
        for i in range(n_instances):
            dists[i, preds[i]] = 1
        return dists

    def _score(self, X, y=None):
        raise NotImplementedError

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances,n_dimensions,series_length)) or pd.Dataframe.
            Time series instances to predict their cluster indexes. If data is not
            equal length a pd.Dataframe given, if another other type of data a
            np.ndarray given.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        raise NotImplementedError

    def _fit(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances,n_dimensions,series_length)) or pd.Dataframe.
            Training time series instances to cluster. If data is not equal length a
            pd.Dataframe given, if another other type of data a np.ndarray given.

        Returns
        -------
        self:
            Fitted estimator.
        """
        raise NotImplementedError

    def _check_capabilities(self, missing: bool, multivariate: bool, unequal: bool):
        """Check the capabilities of the clusterer matches input data requirements.

        Parameters
        ----------
        missing : boolean
            Defines if the data has missing value. True if data has missing values,
            False if no missing.
        multivariate : boolean
            Defines if the data is multivariate. True if data is multivariate, False
            if the data is univariate
        unequal : boolean
            Defines if the data is unequal length. True if data is unequal length,
            False if the data is equal length.

        Raises
        ------
        ValueError
            if the capabilities in self._tags do not handle the data.
        """
        allow_multivariate = self.get_tag("capability:multivariate")
        allow_missing = self.get_tag("capability:missing_values")
        allow_unequal = self.get_tag("capability:unequal_length")
        if missing and not allow_missing:
            raise ValueError(
                "The data has missing values, this clusterer cannot handle missing "
                "values"
            )
        if multivariate and not allow_multivariate:
            raise ValueError(
                "X must be univariate, this clusterer cannot deal with "
                "multivariate input."
            )
        if unequal and not allow_unequal:
            raise ValueError(
                "The data has unequal length series, this clusterer cannot handle "
                "unequal length series"
            )

    @staticmethod
    def _initial_conversion(X: Any) -> TimeSeriesInstances:
        """Format data as valid panel mtype of the data.

        Parameters
        ----------
        X: Any
            Data to convert to panel mtype.

        Returns
        -------
        X: np.ndarray (at least 2d) or pd.Dataframe or List[pd.Dataframe]
            Converted X.
        """
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return X

    def _check_clusterer_input(
        self, X: TimeSeriesInstances, enforce_min_instances: int = 1
    ) -> TimeSeriesInstances:
        """Validate the input and prepare for _fit.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances,n_dimensions,series_length)) or nested pd.DataFrame (
            n_instances,n_dimensions).
            Training time series instances to cluster.

        Returns
        -------
        X : np.ndarray (3d of shape (n_instances,n_dimensions,series_length)) or
            pd.Dataframe (n_instances,n_dimensions).
            Converted X ready for _fit.

        Raises
        ------
        ValueError
            If y or X is invalid input data type, or there is not enough data.
        """
        X = self._initial_conversion(X)

        ALLOWED_SCITYPES = [
            "Panel",
        ]
        FORBIDDEN_MTYPES = []

        mtypes_messages = []
        for scitype in ALLOWED_SCITYPES:
            mtypes = set(scitype_to_mtype(scitype))
            mtypes = list(mtypes.difference(FORBIDDEN_MTYPES))
            mtypes_messages.append(f"For {scitype} scitype: {mtypes}")

        X_metadata_required = [
            "n_instances",
            "has_nans",
            "is_univariate",
            "is_equal_length",
        ]
        X_valid, _, X_metadata = check_is_scitype(
            X, scitype=ALLOWED_SCITYPES, return_metadata=X_metadata_required
        )
        if not X_valid:
            raise TypeError(
                "X must be in a sktime compatible format, of scitype: "
                f"{', '.join(ALLOWED_SCITYPES)}. "
                "For instance a pandas.DataFrame must have a 2-level MultiIndex. "
                "In case of numpy array, it must be "
                "a 3D array as (num_instance, num_vars, series). "
                "If you think X is already is an sktime supported input format, "
                "run `sktime.datatypes.check_raise(X, MTYPE)` to diagnose the error, "
                "where MTYPE is the string of the type specification you want for X. "
                "Possible mtype specification strings are as follows: "
                f"{', '.join(mtypes_messages)}"
            )

        n_cases = X_metadata["n_instances"]
        if n_cases < enforce_min_instances:
            raise ValueError(
                f"Minimum number of cases required is {enforce_min_instances} but X "
                f"has : {n_cases}"
            )
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        self._check_capabilities(missing, multivariate, unequal)
        return convert_to(
            X,
            to_type=self.get_tag("X_inner_mtype"),
            as_scitype="Panel",
        )
