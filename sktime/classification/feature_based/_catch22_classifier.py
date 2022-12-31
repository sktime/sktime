# -*- coding: utf-8 -*-
"""Catch22 Classifier.

Pipeline classifier using the Catch22 transformer and an estimator.
"""

__author__ = ["MatthewMiddlehurst", "RavenRudi", "fkiraly"]
__all__ = ["Catch22Classifier"]

from sklearn.ensemble import RandomForestClassifier

from sktime.base._base import _clone_estimator
from sktime.classification._delegate import _DelegatedClassifier
from sktime.pipeline import make_pipeline
from sktime.transformations.panel.catch22 import Catch22


class Catch22Classifier(_DelegatedClassifier):
    """Canonical Time-series Characteristics (catch22) classifier.

    This classifier simply transforms the input data using the Catch22 [1]
    transformer and builds a provided estimator using the transformed data.

    Shorthand for the pipeline `Catch22(outlier_norm, replace_nans) * estimator`

    Parameters
    ----------
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features, which can take a
        while to process for large values.
    replace_nans : bool, optional, default=True
        Replace NaN or inf values from the Catch22 transform with 0.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestClassifier(n_estimators=200)
    n_jobs : int, optional, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, optional, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    estimator_ : ClassifierPipeline
        Catch22Classifier as a ClassifierPipeline, fitted to data internally

    See Also
    --------
    Catch22

    Notes
    -----
    Authors `catch22ForestClassifier <https://github.com/chlubba/sktime-catch22>`_.

    For the Java version, see `tsml <https://github.com/uea-machine-learning/tsml/blob
    /master/src/main/java/tsml/classifiers/hybrids/Catch22Classifier.java>`_.

    References
    ----------
    .. [1] Lubba, Carl H., et al. "catch22: Canonical time-series characteristics."
        Data Mining and Knowledge Discovery 33.6 (2019): 1821-1852.
        https://link.springer.com/article/10.1007/s10618-019-00647-x

    Examples
    --------
    >>> from sktime.classification.feature_based import Catch22Classifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = Catch22Classifier(
    ...     estimator=RandomForestClassifier(n_estimators=5),
    ...     outlier_norm=True,
    ... )
    >>> clf.fit(X_train, y_train)
    Catch22Classifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "classifier_type": "feature",
    }

    def __init__(
        self,
        outlier_norm=False,
        replace_nans=True,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        super(Catch22Classifier, self).__init__()

        transformer = Catch22(
            outlier_norm=self.outlier_norm, replace_nans=self.replace_nans
        )

        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=200)

        estimator = _clone_estimator(estimator, random_state)

        m = getattr(estimator, "n_jobs", None)
        if m is not None:
            estimator.n_jobs = self._threads_to_use

        self.estimator_ = make_pipeline(transformer, estimator)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {
                "estimator": RandomForestClassifier(n_estimators=10),
                "outlier_norm": True,
            }

        from sklearn.dummy import DummyClassifier

        param1 = {"estimator": RandomForestClassifier(n_estimators=2)}
        param2 = {
            "estimator": DummyClassifier(),
            "outlier_norm": True,
            "replace_nans": False,
            "random_state": 42,
        }

        return [param1, param2]
