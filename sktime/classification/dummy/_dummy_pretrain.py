"""Dummy classifier with pretraining, for testing and baseline comparisons."""

__author__ = ["srupat"]
__all__ = ["DummyPretrainClassifier"]

import numpy as np

from sktime.classification.base import BaseClassifier


class DummyPretrainClassifier(BaseClassifier):
    """Dummy classifier that predicts a class prior, informed by pretraining.

    Ignores the input features, and predicts the same class probabilities for
    all instances.

    In ``pretrain``, counts how often each class label occurs in the labels
    passed to ``pretrain``. ``pretrain`` can also be called without labels,
    in which case only the number of pretraining instances is recorded.
    Further calls of ``pretrain`` add to the counts, instead of replacing them.

    In ``fit``, counts how often each class label occurs in the labels passed to
    ``fit``. The predicted class probabilities are the class frequencies in
    ``fit``, where the class counts from ``pretrain`` are added as
    pseudo-counts, weighted by ``pretrain_weight``. Only the classes seen in
    ``fit`` are predicted, pretraining counts of other classes are ignored.

    Without pretraining, this is the same as the ``"prior"`` strategy of
    ``DummyClassifier``.

    Useful as:

    - test case for the pretraining API of classifiers
    - simple baseline for classification with pretraining

    Parameters
    ----------
    pretrain_weight : float, default=1.0
        Weight of the class counts from ``pretrain``, relative to the class
        counts from ``fit``. Must be non-negative.
        If 0, pretraining has no effect on predictions.

    Attributes
    ----------
    n_pretrain_instances_ : int
        Number of instances seen in all calls of ``pretrain``
        (set after ``pretrain``)
    pretrain_class_counts_ : dict
        Number of occurrences of each class label in the labels seen in all
        calls of ``pretrain``, keys are class labels, values are counts.
        Empty if ``pretrain`` was called without labels (set after ``pretrain``)
    class_prior_ : np.ndarray of shape (n_classes,)
        Predicted class probabilities, in the order of ``classes_``
        (set after ``fit``)

    Examples
    --------
    >>> from sktime.classification.dummy import DummyPretrainClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = DummyPretrainClassifier()
    >>> classifier.pretrain(X_test, y_test)
    DummyPretrainClassifier()
    >>> classifier.fit(X_train, y_train)
    DummyPretrainClassifier()
    >>> y_pred_proba = classifier.predict_proba(X_test)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["srupat"],
        "maintainers": ["srupat"],
        # estimator type
        # --------------
        "X_inner_mtype": ["df-list", "numpy3D"],
        "capability:missing_values": True,
        "capability:unequal_length": True,
        "capability:multivariate": True,
        "capability:predict_proba": True,
        "capability:pretrain": True,
        "pretrain:fitted_params": ["n_pretrain_instances_", "pretrain_class_counts_"],
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, pretrain_weight=1.0):
        self.pretrain_weight = pretrain_weight
        super().__init__()

    def _pretrain(self, X, y=None):
        """Count class labels of the pretraining data, first pretraining batch.

        Parameters
        ----------
        X : list of pd.DataFrame or 3D np.ndarray
            pretraining instances, only the number of instances is used
        y : 1D np.ndarray of shape (n_instances,), or None
            class labels of the pretraining data

        Returns
        -------
        self : reference to self
        """
        self.n_pretrain_instances_ = len(X)
        self.pretrain_class_counts_ = self._count_labels(y)
        return self

    def _pretrain_update(self, X, y=None):
        """Add class label counts of further pretraining data.

        Parameters
        ----------
        X : list of pd.DataFrame or 3D np.ndarray
            pretraining instances, only the number of instances is used
        y : 1D np.ndarray of shape (n_instances,), or None
            class labels of the pretraining data

        Returns
        -------
        self : reference to self
        """
        self.n_pretrain_instances_ += len(X)
        for label, count in self._count_labels(y).items():
            prior_count = self.pretrain_class_counts_.get(label, 0)
            self.pretrain_class_counts_[label] = prior_count + count
        return self

    def _fit(self, X, y):
        """Fit the class prior, using pretraining counts as pseudo-counts.

        Parameters
        ----------
        X : list of pd.DataFrame or 3D np.ndarray
            training instances, ignored
        y : 1D np.ndarray of shape (n_instances,)
            class labels of the training data

        Returns
        -------
        self : reference to self
        """
        fit_counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)

        pretrain_class_counts = getattr(self, "pretrain_class_counts_", {})
        pretrain_counts = np.array(
            [pretrain_class_counts.get(c, 0) for c in self.classes_], dtype=float
        )

        counts = fit_counts + self.pretrain_weight * pretrain_counts
        self.class_prior_ = counts / counts.sum()
        return self

    def _predict(self, X) -> np.ndarray:
        """Predict the most probable class for all instances in X.

        Parameters
        ----------
        X : list of pd.DataFrame or 3D np.ndarray
            instances to predict labels for, only the number of instances is used

        Returns
        -------
        y : 1D np.ndarray of shape (n_instances,), predicted class labels
        """
        most_probable = self.classes_[np.argmax(self.class_prior_)]
        return np.repeat(most_probable, len(X))

    def _predict_proba(self, X) -> np.ndarray:
        """Predict the class prior for all instances in X.

        Parameters
        ----------
        X : list of pd.DataFrame or 3D np.ndarray
            instances to predict probabilities for, only the number of instances
            is used

        Returns
        -------
        y : 2D np.ndarray of shape (n_instances, n_classes)
            predicted class probabilities, columns in the order of ``classes_``
        """
        return np.tile(self.class_prior_, (len(X), 1))

    @staticmethod
    def _count_labels(y):
        """Count occurrences of each class label in y, empty dict if y is None."""
        if y is None:
            return {}
        labels, counts = np.unique(y, return_counts=True)
        return dict(zip(labels, counts.tolist()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        return [{}, {"pretrain_weight": 0.5}]
