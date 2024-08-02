"""TSFresh Classifier.

Pipeline classifier using the TSFresh transformer and an estimator.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["TSFreshClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.tsfresh import (
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor,
)
from sktime.utils.warnings import warn


class TSFreshClassifier(BaseClassifier):
    """Time Series Feature Extraction based on Scalable Hypothesis Tests classifier.

    This classifier simply transforms the input data using the TSFresh [1]
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    default_fc_parameters : str, default="efficient"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    relevant_feature_extractor : bool, default=False
        Remove irrelevant features using the FRESH algorithm.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    verbose : int, default=0
        level of output printed to the console (for information only)
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``-1`` means using all processors.
    chunksize : int or None, default=None
        Number of series processed in each parallel TSFresh job, should be optimised
        for efficient parallelisation.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.

    See Also
    --------
    TSFreshFeatureExtractor, TSFreshRelevantFeatureExtractor

    References
    ----------
    .. [1] Christ, Maximilian, et al. "Time series feature extraction on basis of
        scalable hypothesis tests (tsfresh-a python package)." Neurocomputing 307
        (2018): 72-77.
        https://www.sciencedirect.com/science/article/pii/S0925231218304843

    Examples
    --------
    >>> from sktime.classification.feature_based import TSFreshClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True) # doctest: +SKIP
    >>> clf = TSFreshClassifier(
    ...     estimator=RandomForestClassifier(n_estimators=5),
    ...     default_fc_parameters="efficient",
    ... ) # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    TSFreshClassifier(...)
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["MatthewMiddlehurst"],
        "python_version": "<3.10",
        "python_dependencies": "tsfresh",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "feature",
    }

    def __init__(
        self,
        default_fc_parameters="efficient",
        relevant_feature_extractor=True,
        estimator=None,
        verbose=0,
        n_jobs=1,
        chunksize=None,
        random_state=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.relevant_feature_extractor = relevant_feature_extractor
        self.estimator = estimator

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

        self._transformer = None
        self._estimator = None
        self._return_majority_class = False
        self._majority_class = 0

        super().__init__()

    def _fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self._transformer = (
            TSFreshRelevantFeatureExtractor(
                default_fc_parameters=self.default_fc_parameters,
                n_jobs=self._threads_to_use,
                chunksize=self.chunksize,
            )
            if self.relevant_feature_extractor
            else TSFreshFeatureExtractor(
                default_fc_parameters=self.default_fc_parameters,
                n_jobs=self._threads_to_use,
                chunksize=self.chunksize,
            )
        )
        self._estimator = _clone_estimator(
            (
                RandomForestClassifier(n_estimators=200)
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        if self.verbose < 2:
            self._transformer.show_warnings = False
            if self.verbose < 1:
                self._transformer.disable_progressbar = True

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._threads_to_use

        X_t = self._transformer.fit_transform(X, y)
        self._Xt_colnames = X_t.columns

        if X_t.shape[1] == 0:
            warn(
                "TSFresh has extracted no features from the data. Returning the "
                "majority class in predictions. Setting "
                "relevant_feature_extractor=False will keep all features.",
                UserWarning,
                stacklevel=2,
            )

            self._return_majority_class = True
            self._majority_class = np.argmax(np.unique(y, return_counts=True)[1])
        else:
            self._estimator.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        if self._return_majority_class:
            return np.full(X.shape[0], self.classes_[self._majority_class])

        X_t = self._transformer.transform(X)
        X_t = X_t.reindex(self._Xt_colnames, axis=1, fill_value=0)
        return self._estimator.predict(X_t)

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        if self._return_majority_class:
            dists = np.zeros((X.shape[0], self.n_classes_))
            dists[:, self._majority_class] = 1
            return dists

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            X_t = self._transformer.transform(X)
            X_t = X_t.reindex(self._Xt_colnames, axis=1, fill_value=0)
            preds = self._estimator.predict(X_t)
            for i in range(0, X.shape[0]):
                dists[i, self._class_dictionary[preds[i]]] = 1
            return dists

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        if parameter_set == "results_comparison":
            return {
                "estimator": RandomForestClassifier(n_estimators=10),
                "default_fc_parameters": "minimal",
                "relevant_feature_extractor": False,
            }
        else:
            return {
                "estimator": RandomForestClassifier(n_estimators=2),
                "default_fc_parameters": "minimal",
                "relevant_feature_extractor": False,
            }
