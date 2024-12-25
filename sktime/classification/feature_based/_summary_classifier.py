"""Summary Classifier.

Pipeline classifier using the basic summary statistics and an estimator.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["SummaryClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.series.summarize import SummaryTransformer


class SummaryClassifier(BaseClassifier):
    """Summary statistic classifier.

    This classifier simply transforms the input data using the SummaryTransformer
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    summary_functions : str, list, tuple, default=("mean", "std", "min", "max")
        Either a string, or list or tuple of strings indicating the pandas
        summary functions that are used to summarize each column of the dataset.
        Must be one of ("mean", "min", "max", "median", "sum", "skew", "kurt",
        "var", "std", "mad", "sem", "nunique", "count").
    summary_quantiles : str, list, tuple or None, default=(0.25, 0.5, 0.75)
        Optional list of series quantiles to calculate. If None, no quantiles
        are calculated.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes)
        Holds the label for each class.

    See Also
    --------
    SummaryTransformer

    Examples
    --------
    >>> from sktime.classification.feature_based import SummaryClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = SummaryClassifier(estimator=RandomForestClassifier(n_estimators=5))
    >>> clf.fit(X_train, y_train)
    SummaryClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["MatthewMiddlehurst"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "feature",
    }

    def __init__(
        self,
        summary_functions=("mean", "std", "min", "max"),
        summary_quantiles=(0.25, 0.5, 0.75),
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.summary_functions = summary_functions
        self.summary_quantiles = summary_quantiles
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._transformer = None
        self._estimator = None
        self._transform_atts = 0

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
        self._transformer = SummaryTransformer(
            summary_function=self.summary_functions,
            quantiles=self.summary_quantiles,
        )

        self._estimator = _clone_estimator(
            (
                RandomForestClassifier(n_estimators=200)
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._threads_to_use

        X_t = self._transformer.fit_transform(X, y)

        if X_t.shape[0] > len(y):
            X_t = X_t.to_numpy().reshape((len(y), -1))
            self._transform_atts = X_t.shape[1]

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
        X_t = self._transformer.transform(X)

        if X_t.shape[1] < self._transform_atts:
            X_t = X_t.to_numpy().reshape((-1, self._transform_atts))

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
        X_t = self._transformer.transform(X)

        if X_t.shape[1] < self._transform_atts:
            X_t = X_t.to_numpy().reshape((-1, self._transform_atts))

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_t)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
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
            return {"estimator": RandomForestClassifier(n_estimators=10)}
        else:
            return {
                "estimator": RandomForestClassifier(n_estimators=2),
                "summary_functions": ("mean", "min", "max"),
            }
