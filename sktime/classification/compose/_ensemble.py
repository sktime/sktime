# -*- coding: utf-8 -*-
"""
Configurable time series ensembles
"""
__author__ = ["Markus Löning", "Ayushmaan Seth"]
__all__ = ["ComposableTimeSeriesForestClassifier"]

from warnings import warn
import numpy as np
import numbers
from joblib import Parallel
from joblib import delayed

from sklearn.ensemble._base import _partition_estimators
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.pipeline import Pipeline
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.ensemble._forest import _get_n_samples_bootstrap
from sktime.transformations.panel.summarize import (
    RandomIntervalFeatureExtractor,
)
from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation.panel import check_X, check_X_y
from sktime.classification.base import BaseClassifier
from sktime.series_as_features.base.estimators._ensemble import BaseTimeSeriesForest


class ComposableTimeSeriesForestClassifier(BaseTimeSeriesForest, BaseClassifier):
    """Time-Series Forest Classifier.

    @article{DENG2013142,
        title = {A time series forest for classification and feature extraction},
        journal = {Information Sciences},
        volume = {239},
        pages = {142 - 153},
        year = {2013},
        issn = {0020-0255},
        doi = {https://doi.org/10.1016/j.ins.2013.02.030},
        url = {http://www.sciencedirect.com/science/article/pii/S0020025513001473},
        author = {Houtao Deng and George Runger and Eugene Tuv and Martyanov Vladimir},
        keywords = {Decision tree, Ensemble, Entrance gain, Interpretability,
                    Large margin, Time series classification}
    }

    A time series forest is a meta estimator and an adaptation of the random
    forest for time-series/panel data that fits a number of decision tree
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
    criterion : string, optional (default="entropy")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific. Default is "entropy"
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
    min_impurity_split : float or None, (default=None)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
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
    """

    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
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
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
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

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

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
            self.estimator_ = Pipeline(steps)

        else:
            # else check given estimator is a pipeline with prior
            # transformations and final decision tree
            if not isinstance(self.estimator, Pipeline):
                raise ValueError("`estimator` must be pipeline with transforms.")
            if not isinstance(self.estimator.steps[-1][1], DecisionTreeClassifier):
                raise ValueError(
                    "Last step in `estimator` must be DecisionTreeClassifier."
                )
            self.estimator_ = self.estimator

        # Set parameters according to naming in pipeline
        estimator_params = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "min_impurity_split": self.min_impurity_split,
        }
        final_estimator = self.estimator_.steps[-1][0]
        self.estimator_params = {
            f"{final_estimator}__{pname}": pval
            for pname, pval in estimator_params.items()
        }

        # Set renamed estimator parameters
        for pname, pval in self.estimator_params.items():
            self.__setattr__(pname, pval)

    def predict(self, X):
        """
        Predict class for X.
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
        """
        Predict class log-probabilities for X.
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

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the
        same
        class in a leaf.
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
        """Compute out-of-bag score"""
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

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
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
