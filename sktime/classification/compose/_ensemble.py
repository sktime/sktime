# -*- coding: utf-8 -*-
"""Configurable time series ensembles."""
__author__ = ["mloning", "AyushmaanSeth", "fkiraly"]
__all__ = ["ComposableTimeSeriesForestClassifier"]

import numbers
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import (
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
)
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets

from sktime.base import _HeterogenousMetaEstimator
from sktime.classification.base import BaseClassifier
from sktime.series_as_features.base.estimators._ensemble import BaseTimeSeriesForest
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation.panel import check_X, check_X_y


class ComposableTimeSeriesForestClassifier(BaseTimeSeriesForest, BaseClassifier):
    """Time Series Forest Classifier as described in [1]_.

    A time series forest is an adaptation of the random
    forest for time-series data. It that fits a number of decision tree
    classifiers on various sub-samples of a transformed dataset and uses
    averaging to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original input sample size
    but the samples are drawn with replacement if `bootstrap=True` (default).

    Parameters
    ----------
    estimator : Pipeline
        A pipeline consisting of series-to-tabular transformations
        and a decision tree classifier as final estimator.
    n_estimators : integer, optional (default=200)
        The number of trees in the forest.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.
    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
        None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).
    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).
    n_columns : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    feature_importances_ : data frame of shape = [n_timepoints, n_features]
        The normalised feature values at each time index of
        the time series forest
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------
    .. [1] Deng et. al, A time series forest for classification and feature extraction,
    Information Sciences, 239:2013.
    """

    _tags = {
        "X_inner_mtype": "nested_univ",  # nested pd.DataFrame
    }

    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):

        self.estimator = estimator

        # Assign values, even though passed on to base estimator below,
        # necessary here for cloning
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = max_samples

        # Pass on params.
        super(ComposableTimeSeriesForestClassifier, self).__init__(
            base_estimator=None,
            n_estimators=n_estimators,
            estimator_params=None,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )
        BaseClassifier.__init__(self)

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit(self, X, y, **kwargs):
        """Wrap fit to call BaseClassifier.fit.

        This is a fix to get around the problem with multiple inheritance. The
        problem is that if we just override _fit, this class inherits the fit from
        the sklearn class BaseTimeSeriesForest. This is the simplest solution,
        albeit a little hacky.
        """
        return BaseClassifier.fit(self, X=X, y=y, **kwargs)

    def predict(self, X, **kwargs) -> np.ndarray:
        """Wrap predict to call BaseClassifier.predict."""
        return BaseClassifier.predict(self, X=X, **kwargs)

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        """Wrap predict_proba to call BaseClassifier.predict_proba."""
        return BaseClassifier.predict_proba(self, X=X, **kwargs)

    def _fit(self, X, y):
        BaseTimeSeriesForest._fit(self, X=X, y=y)

    def _validate_estimator(self):

        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError(
                "n_estimators must be an integer, "
                "got {0}.".format(type(self.n_estimators))
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than zero, "
                "got {0}.".format(self.n_estimators)
            )

        # Set base estimator
        if self.estimator is None:
            # Set default time series forest
            features = [np.mean, np.std, _slope]
            steps = [
                (
                    "transform",
                    RandomIntervalFeatureExtractor(
                        n_intervals="sqrt",
                        features=features,
                        random_state=self.random_state,
                    ),
                ),
                ("clf", DecisionTreeClassifier(random_state=self.random_state)),
            ]
            self._estimator = Pipeline(steps)

        else:
            # else check given estimator is a pipeline with prior
            # transformations and final decision tree
            if not isinstance(self.estimator, Pipeline):
                raise ValueError("`estimator` must be pipeline with transforms.")
            if not isinstance(self.estimator.steps[-1][1], DecisionTreeClassifier):
                raise ValueError(
                    "Last step in `estimator` must be DecisionTreeClassifier."
                )
            self._estimator = self.estimator

        # Set parameters according to naming in pipeline
        estimator_params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
        }
        final_estimator = self._estimator.steps[-1][0]
        self.estimator_params = {
            f"{final_estimator}__{pname}": pval
            for pname, pval in estimator_params.items()
        }

        # Set renamed estimator parameters
        for pname, pval in self.estimator_params.items():
            self.__setattr__(pname, pval)

    def _predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_), dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                    np.argmax(proba[k], axis=1), axis=0
                )

            return predictions

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape (n_samples, n_classes), or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

    def _predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the
        same class in a leaf.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        # Check data
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)

        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(e.predict_proba)(X) for e in self.estimators_
        )

        return np.sum(all_proba, axis=0) / len(self.estimators_)

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score."""
        check_X_y(X, y)
        check_X(X, enforce_univariate=True)

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = [
            np.zeros((n_samples, n_classes_[k])) for k in range(self.n_outputs_)
        ]

        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.max_samples)

        for estimator in self.estimators_:
            final_estimator = estimator.steps[-1][1]
            unsampled_indices = _generate_unsampled_indices(
                final_estimator.random_state, n_samples, n_samples_bootstrap
            )
            p_estimator = estimator.predict_proba(X.iloc[unsampled_indices, :])

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn(
                    "Some inputs do not have OOB scores. "
                    "This probably means too few trees were used "
                    "to compute any reliable oob estimates."
                )

            decision = predictions[k] / predictions[k].sum(axis=1)[:, np.newaxis]
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] == np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_

    # TODO - Implement this abstract method properly.
    def _set_oob_score_and_attributes(self, X, y):
        raise NotImplementedError("Not implemented.")

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(
                y[:, k], return_inverse=True
            )
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ("balanced", "balanced_subsample")
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError(
                        "Valid presets for class_weight include "
                        '"balanced" and "balanced_subsample".'
                        'Given "%s".' % self.class_weight
                    )
                if self.warm_start:
                    warn(
                        'class_weight presets "balanced" or '
                        '"balanced_subsample" are '
                        "not recommended for warm_start if the fitted data "
                        "differs from the full dataset. In order to use "
                        '"balanced" weights, use compute_class_weight '
                        '("balanced", classes, y). In place of y you can use '
                        "a large enough sample of the full training set "
                        "target to properly estimate the class frequency "
                        "distributions. Pass the resulting weights as the "
                        "class_weight parameter."
                    )

            if self.class_weight != "balanced_subsample" or not self.bootstrap:
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight, y_original)

        return y, expanded_class_weight

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
        return {"n_estimators": 2}


class WeightedEnsembleClassifier(_HeterogenousMetaEstimator, BaseClassifier):
    """Weighted ensemble of classifiers with fittable ensemble weight.

    Produces a probabilistic prediction which is the weighted average of
    predictions of individual classifiers.
    Classifier with name `name` has ensemble weight in `weights_[name]`.
    `weights_` is fitted in `fit`, if `weights` is a scalar, otherwise fixed.

    If `weights` is a scalar, empirical training loss is computed for each classifier.
    In this case, ensemble weights of classifier is empirical loss,
    to the power of `weights` (a scalar).

    The evaluation for the empirical training loss can be selected
    through the `metric` and `metric_type` parameters.

    The in-sample empirical training loss is computed in-sample or out-of-sample,
    depending on the `cv` parameter. None = in-sample; other = cross-validated oos.

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
        clones of classifies in `classifiers` which are fitted in the ensemble
        is always in (str, classifier) format, even if `classifiers` is just a list
        strings not passed in `classifiers` are replaced by unique generated strings
        i-th classifier in `classifier_` is clone of i-th in `classifier`
    weights_ : dict with str being classifier names as in `classifiers_`
        value at key is ensemble weights of classifier with name key
        ensemble weights are fitted in `fit` if `weights` is a scalar

    Examples
    --------
    >>> from sktime.classification.dummy import DummyClassifier
    >>> from sktime.classification.kernel_based import RocketClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = WeightedEnsembleClassifier(
    ...     [DummyClassifier(), RocketClassifier(num_kernels=100)],
    ...     weights=2,
    ... )
    >>> clf.fit(X_train, y_train)
    WeightedEnsembleClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:missing_values": True,
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
    # this must be an iterable of (name: str, estimator) pairs for the default
    _steps_attr = "_classifiers"

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

        super(WeightedEnsembleClassifier, self).__init__()

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
        dists = np.zeros((X.shape[0], self.n_classes_))

        # Call predict proba on each classifier, multiply the probabilities by the
        # classifiers weight then add them to the current HC2 probabilities
        for clf_name, clf in self.classifiers_:
            y_proba = clf.predict_proba(X=X)
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
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
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
        return [params1, params2]
