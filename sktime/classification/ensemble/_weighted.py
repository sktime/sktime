"""Weighted ensemble of classifiers."""

__author__ = ["fkiraly"]
__all__ = ["WeightedEnsembleClassifier"]

import numpy as np
from sklearn.metrics import accuracy_score

from sktime.base import _HeterogenousMetaEstimator
from sktime.classification.base import BaseClassifier


class WeightedEnsembleClassifier(_HeterogenousMetaEstimator, BaseClassifier):
    """Weighted ensemble of classifiers with fittable ensemble weight.

    Produces a probabilistic prediction which is the weighted average of
    predictions of individual classifiers.
    Classifier with name ``name`` has ensemble weight in ``weights_[name]``.
    ``weights_`` is fitted in ``fit``, if ``weights`` is a scalar, otherwise fixed.

    If ``weights`` is a scalar, empirical training loss is computed for each classifier.
    In this case, ensemble weights of classifier is empirical loss,
    to the power of ``weights`` (a scalar).

    The evaluation for the empirical training loss can be selected
    through the ``metric`` and ``metric_type`` parameters.

    The in-sample empirical training loss is computed in-sample or out-of-sample,
    depending on the ``cv`` parameter. None = in-sample; other = cross-validated oos.

    Parameters
    ----------
    classifiers : dict or None, default=None
        Parameters for the ShapeletTransformClassifier module. If None, uses the
        default parameters with a 2 hour transform contract.
    weights : float, or iterable of float, optional, default=None
        if float, ensemble weight for classifier i will be train score to this power
        if iterable of float, must be equal length as classifiers
            ensemble weight for classifier i will be weights[i]
        if None, ensemble weights are equal (uniform average)
    cv : None, int, or sklearn cross-validation object, optional, default=None
        determines whether in-sample or which cross-validated predictions used in fit
        None : predictions are in-sample, equivalent to fit(X, y).predict(X)
        cv : predictions are equivalent to fit(X_train, y_train).predict(X_test)
            where multiple X_train, y_train, X_test are obtained from cv folds
            returned y is union over all test fold predictions
            cv test folds must be non-intersecting
        int : equivalent to cv=KFold(cv, shuffle=True, random_state=x),
            i.e., k-fold cross-validation predictions out-of-sample
            random_state x is taken from self if exists, otherwise x=None
    metric : sklearn metric for computing training score, default=accuracy_score
        only used if weights is a float
    metric_type : str, one of "point" or "proba", default="point"
        type of sklearn metric, point prediction ("point") or probabilistic ("proba")
        if "point", most probable class is passed as y_pred
        if "proba", probability of most probable class is passed as y_pred
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    classifiers_ : list of tuples (str, classifier) of sktime classifiers
        clones of classifies in ``classifiers`` which are fitted in the ensemble
        is always in (str, classifier) format, even if ``classifiers`` is just a list
        strings not passed in ``classifiers`` are replaced by unique generated strings
        i-th classifier in ``classifier_`` is clone of i-th in ``classifier``
    weights_ : dict with str being classifier names as in ``classifiers_``
        value at key is ensemble weights of classifier with name key
        ensemble weights are fitted in ``fit`` if ``weights`` is a scalar

    Examples
    --------
    >>> from sktime.classification.dummy import DummyClassifier
    >>> from sktime.classification.kernel_based import RocketClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train") # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test") # doctest: +SKIP
    >>> clf = WeightedEnsembleClassifier(
    ...     [DummyClassifier(), RocketClassifier(num_kernels=100)],
    ...     weights=2,
    ... ) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    WeightedEnsembleClassifier(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "fkiraly",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "X_inner_mtype": [
            "pd-multiindex",
            "df-list",
            "nested_univ",
            "numpy3D",
        ],
    }

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_classifiers"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "classifiers_"

    def __init__(
        self,
        classifiers,
        weights=None,
        cv=None,
        metric=None,
        metric_type="point",
        random_state=None,
    ):
        self.classifiers = classifiers
        self.weights = weights
        self.cv = cv
        self.metric = metric
        self.metric_type = metric_type
        self.random_state = random_state

        # make the copies that are being fitted
        self.classifiers_ = self._check_estimators(
            self.classifiers, cls_type=BaseClassifier
        )

        # pass on random state
        for _, clf in self.classifiers_:
            params = clf.get_params()
            if "random_state" in params and params["random_state"] is None:
                clf.set_params(random_state=random_state)

        if weights is None:
            self.weights_ = {x[0]: 1 for x in self.classifiers_}
        elif isinstance(weights, (float, int)):
            self.weights_ = dict()
        elif isinstance(weights, dict):
            self.weights_ = {x[0]: weights[x[0]] for x in self.classifiers_}
        else:
            self.weights_ = {x[0]: weights[i] for i, x in enumerate(self.classifiers_)}

        if metric is None:
            self._metric = accuracy_score
        else:
            self._metric = metric

        super().__init__()

        # set property tags based on tags of components
        ests = self.classifiers_
        self._anytagis_then_set("capability:multivariate", False, True, ests)
        self._anytagis_then_set("capability:missing_values", False, True, ests)

    @property
    def _classifiers(self):
        return self._get_estimator_tuples(self.classifiers, clone_ests=False)

    @_classifiers.setter
    def _classifiers(self, value):
        self.classifiers = value

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : 1D np.array of int, of shape [n_instances] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.
        """
        # if weights are fixed, we only fit
        if not isinstance(self.weights, (float, int)):
            for _, classifier in self.classifiers_:
                classifier.fit(X=X, y=y)
        # if weights are calculated by training loss, we fit_predict and evaluate
        else:
            exponent = self.weights
            for clf_name, clf in self.classifiers_:
                train_probs = clf.fit_predict_proba(X=X, y=y, cv=self.cv)
                train_preds = clf.classes_[np.argmax(train_probs, axis=1)]
                if self.metric_type == "proba":
                    for i in range(len(train_preds)):
                        train_preds[i] = train_probs[i, np.argmax(train_probs[i, :])]
                metric = self._metric
                self.weights_[clf_name] = metric(y, train_preds) ** exponent

        return self

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        dists = None

        # Call predict proba on each classifier, multiply the probabilities by the
        # classifiers weight then add them to the current HC2 probabilities
        for clf_name, clf in self.classifiers_:
            y_proba = clf.predict_proba(X=X)
            if dists is None:
                dists = y_proba * self.weights_[clf_name]
            else:
                dists += y_proba * self.weights_[clf_name]

        # Make each instances probability array sum to 1 and return
        y_proba = dists / dists.sum(axis=1, keepdims=True)

        return y_proba

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
        from sktime.classification.dummy import DummyClassifier
        from sktime.utils.dependencies import _check_soft_dependencies

        params0 = {"classifiers": [DummyClassifier()]}

        if _check_soft_dependencies("numba", severity="none"):
            from sktime.classification.distance_based import (
                KNeighborsTimeSeriesClassifier,
            )
            from sktime.classification.kernel_based import RocketClassifier

            params1 = {
                "classifiers": [
                    KNeighborsTimeSeriesClassifier.create_test_instance(),
                    RocketClassifier.create_test_instance(),
                ],
                "weights": [42, 1],
            }

            params2 = {
                "classifiers": [
                    KNeighborsTimeSeriesClassifier.create_test_instance(),
                    RocketClassifier.create_test_instance(),
                ],
                "weights": 2,
                "cv": 3,
            }
            return [params0, params1, params2]
        else:
            return params0
