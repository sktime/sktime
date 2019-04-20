from warnings import warn
from warnings import catch_warnings
from warnings import simplefilter
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.ensemble.forest import ForestClassifier
from sklearn.ensemble.forest import MAX_INT
from sklearn.ensemble.forest import _generate_sample_indices
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble.base import _partition_estimators, _set_random_states, clone
from sklearn.utils._joblib import Parallel, delayed
from sklearn.tree._tree import DOUBLE
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import class_distribution
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, cross_val_predict
from sktime.transformers.series_to_series import DerivativeSlopeTransformer
from ..pipeline import TSPipeline
from ..transformers.series_to_tabular import RandomIntervalFeatureExtractor
from ..utils.time_series import time_series_slope
import os
from .time_series_neighbors import KNeighborsTimeSeriesClassifier as KNNTSC
from ..distances.elastic_cython import dtw_distance as dtw_c, wdtw_distance as wdtw_c, ddtw_distance as ddtw_c, \
    wddtw_distance as wddtw_c, lcss_distance as lcss_c, erp_distance as erp_c, msm_distance as msm_c
from itertools import product
import time


__all__ = ["TimeSeriesForestClassifier", "ElasticEnsemble"]


class TimeSeriesForestClassifier(ForestClassifier):
    """Time-Series Forest Classifier.

    A time series forest is a meta estimator and an adaptation of the random forest
    for time-series/panel data that fits a number of decision tree classifiers on
    various sub-samples of a transformed dataset and uses averaging to improve the
    predictive accuracy and control over-fitting. The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if `bootstrap=True` (default).

    Parameters
    ----------
    base_estimator : TSPipeline
        A pipeline consisting of series-to-tabular transformers
        and a decision tree classifier as final estimator.
    n_estimators : integer, optional (default=100)
        The number of trees in the forest.
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.
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
    max_features : int, float, string or None, optional (default="auto")
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
    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.
    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
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
        new forest. See :term:`the Glossary <warm_start>`.
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
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=500,
                 criterion='entropy',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 check_input=True):

        if base_estimator is None:
            features = [np.mean, np.std, time_series_slope]
            steps = [('transform', RandomIntervalFeatureExtractor(n_intervals='sqrt', features=features)),
                     ('clf', DecisionTreeClassifier())]
            base_estimator = TSPipeline(steps)

        elif not isinstance(base_estimator, TSPipeline):
            raise ValueError('Base estimator must be pipeline with transforms.')
        elif not isinstance(base_estimator.steps[-1][1], DecisionTreeClassifier):
            raise ValueError('Last step in base estimator pipeline must be DecisionTreeClassifier.')

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.check_input = check_input

        # Pass on params.
        super(TimeSeriesForestClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight
        )

        # Assign random state to pipeline.
        base_estimator.set_params(**{'random_state': random_state, 'check_input': False})

    def fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """

        # Validate or convert input data
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1] if X.ndim == 2 else 1

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: for standard random forests, the threading
            # backend is preferred as the Cython code for fitting the trees
            # is internally releasing the Python GIL making threading more
            # efficient than multiprocessing in that case. However, in this case,
            # for fitting pipelines in parallel, multiprocessing is more efficient.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the same
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
        check_is_fitted(self, 'estimators_')

        # Check data
        if self.check_input:
            X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(delayed(e.predict_proba)(X) for e in self.estimators_)

        all_proba = np.sum(all_proba, axis=0) / len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def _validate_X_predict(self, X):
        n_features = X.shape[1] if X.ndim == 2 else 1
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))
        return X

    def apply(self, X):
        raise NotImplementedError()

    def decision_path(self, X):
        raise NotImplementedError()

    @property
    def feature_importances_(self):
        raise NotImplementedError

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""
        if X.ndim == 1:
            X = pd.DataFrame(X)

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = [np.zeros((n_samples, n_classes_[k]))
                       for k in range(self.n_outputs_)]

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict_proba(X.iloc[unsampled_indices, :])


            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Adapted to handle pipelines as `base_estimators_`.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)

        # Name of final estimator in pipeline.
        final_estimator = estimator.steps[-1][0]
        estimator.set_params(**{f'{final_estimator}__{p}': getattr(self, p)
                                for p in self.estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

def _parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees,
                          verbose=0, class_weight=None):
    """Private function used to fit a single tree in parallel, adjusted for pipeline trees."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    # name of step of final estimator in pipeline
    estimator = tree.steps[-1][0]

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, indices)

        fit_params = {f'{estimator}__sample_weight': curr_sample_weight,
                      f'{estimator}__check_input': True}
        tree.fit(X, y, **fit_params)

    else:
        fit_params = {f'{estimator}__sample_weight': sample_weight,
                      f'{estimator}__check_input': True}
        tree.fit(X, y, **fit_params)

    return tree


class ElasticEnsemble:
    """ The Elastic Ensemble

    An ensemble of elastic nearest neighbor classifiers

    """
    def __init__(
            self,
            distance_measures_to_include='all',
            proportion_of_param_options=1.0,
            proportion_train_in_param_finding=1.0,
            proportion_train_for_test=1.0,
            data_dimension_to_use=0,
            random_seed=0,
            dim_to_use=0,
            verbose=0
    ):
        if distance_measures_to_include == 'all':
            self.distance_measures = [dtw_c, ddtw_c, wdtw_c, wddtw_c, lcss_c, erp_c, msm_c]
        else:
            self.distance_measures = distance_measures_to_include
        self.prop_train_in_param_finding = proportion_train_in_param_finding
        self.prop_of_param_options = proportion_of_param_options
        self.prop_train_for_test = proportion_train_for_test
        self.data_dimension_to_use = data_dimension_to_use
        self.random_seed = random_seed
        self.dim_to_use = dim_to_use
        self.estimators_ = None
        self.train_accs_by_classifier = None
        self.train_preds_by_classifier = None
        self.classes_ = None
        self.verbose = verbose
        self.train = None
        self.constituent_build_times = None

    def fit(self, X, y, **kwargs):

        # Derivative DTW (DDTW) uses the regular DTW algorithm on data that are transformed into derivatives.
        # To increase the efficiency of DDTW we can pre-transform the data into derivatives, and then call the
        # standard DTW algorithm on it, rather than transforming each series every time a distance calculation
        # is made. Please note that using DDTW elsewhere will not benefit from this speed enhancement
        if self.distance_measures.__contains__(ddtw_c) or self.distance_measures.__contains__(wddtw_c):
            der_X = DerivativeSlopeTransformer().transform(X)
            # reshape X for use with the efficient cython distance measures
            der_X = np.array([np.asarray([x]).reshape(len(x), 1) for x in der_X.iloc[:, self.dim_to_use]])
        else:
            der_X = None

        # reshape X for use with the efficient cython distance measures
        X = np.array([np.asarray([x]).reshape(len(x),1) for x in X.iloc[:, self.dim_to_use]])

        self.train_accs_by_classifier = np.zeros(len(self.distance_measures))
        self.train_preds_by_classifier = [None] * len(self.distance_measures)
        self.estimators_ = [None] * len(self.distance_measures)
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        rand = np.random.RandomState(self.random_seed)

        # The default EE uses all training instances for setting parameters, and 100 parameter options per
        # elastic measure. The prop_train_in_param_finding and prop_of_param_options attributes of this class
        # can be used to control this however, using less cases to optimise parameters on the training data
        # and/or using less parameter options.
        #
        # For using less training instances the appropriate number of cases must be sampled from the data.
        # This is achieved through the use of a deterministic StratifiedShuffleSplit
        #
        # For using less parameter options a RandomizedSearchCV is used in place of a GridSearchCV

        param_train_x = None
        der_param_train_x = None
        param_train_y = None

        # If using less cases for parameter optimisation, use the StratifiedShuffleSplit:
        if self.prop_train_in_param_finding < 1:
            if self.verbose > 0:
                print("Restricting training cases for parameter optimisation: ",end="")
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-self.prop_train_in_param_finding, random_state=rand)
            for train_index, test_index in sss.split(X, y):
                param_train_x = X[train_index,:]
                param_train_y = y[train_index]
                if der_X is not None:
                    der_param_train_x = der_X[train_index,:]
                if self.verbose > 0:
                    print("using "+str(len(param_train_x))+" training cases instead of "+str(len(X))+" for parameter optimisation")
        # else, use the full training data for optimising parameters
        else:
            if self.verbose > 0:
                print("Using all training cases for parameter optimisation")
            param_train_x = X
            param_train_y = y
            if der_X is not None:
                der_param_train_x = der_X

        self.constituent_build_times = []

        if self.verbose > 0:
            print("Using "+str(100*self.prop_of_param_options)+" parameter options per measure")
        for dm in range(0, len(self.distance_measures)):
            this_measure = self.distance_measures[dm]

            # uses the appropriate training data as required (either full or smaller sample as per the StratifiedShuffleSplit)
            param_train_to_use = param_train_x
            full_train_to_use = X
            if this_measure is ddtw_c or dm is wddtw_c:
                param_train_to_use = der_param_train_x
                full_train_to_use = der_X
                if this_measure is ddtw_c:
                    this_measure = dtw_c
                elif this_measure is wddtw_c:
                    this_measure = wdtw_c

            start_build_time = time.time()
            if self.verbose > 0:
                if self.distance_measures[dm] is ddtw_c or self.distance_measures[dm] is wddtw_c:
                    print("Currently evaluating "+str(self.distance_measures[dm].__name__)+" (implemented as "+str(this_measure.__name__)+" with pre-transformed derivative data)")
                else:
                    print("Currently evaluating "+str(self.distance_measures[dm].__name__))

            # If 100 parameter options are being considered per measure, use a GridSearchCV
            if self.prop_of_param_options == 1:

                grid = GridSearchCV(
                    estimator= KNNTSC(metric=this_measure, n_neighbors=1, algorithm="brute"),
                    param_grid=ElasticEnsemble._get_100_param_options(self.distance_measures[dm], X),
                    cv=LeaveOneOut(),
                    scoring='accuracy',
                    verbose=self.verbose
                )
                grid.fit(param_train_to_use, param_train_y)

            # Else, used RandomizedSearchCV to randomly sample parameter options for each measure
            else:
                grid = RandomizedSearchCV(
                    estimator=KNNTSC(metric=this_measure, n_neighbors=1, algorithm="brute"),
                    param_distributions=ElasticEnsemble._get_100_param_options(self.distance_measures[dm], X),
                    cv=LeaveOneOut(),
                    scoring='accuracy',
                    n_iter=100 * self.prop_of_param_options,
                    random_state=rand,
                    verbose=self.verbose
                )
                grid.fit(param_train_to_use, param_train_y)

            # once the best parameter option has been estimated on the training data, perform a final pass with this parameter option
            # to get the individual predictions with cross_cal_predict (Note: optimisation potentially possible here if a GridSearchCV
            # was used previously. TO-DO: determine how to extract predictions for the best param option from GridSearchCV)
            best_model = KNNTSC(algorithm="brute", n_neighbors=1, metric=this_measure, metric_params=grid.best_params_['metric_params'])
            preds = cross_val_predict(best_model, full_train_to_use, y, cv=LeaveOneOut())
            acc = accuracy_score(y,preds)

            if self.verbose > 0:
                print("Training accuracy for "+str(self.distance_measures[dm].__name__)+": "+str(acc) + " (with parameter setting: "+str(grid.best_params_['metric_params'])+")")

            # Finally, reset the classifier for this measure and parameter option, ready to be called for test classification
            best_model = KNNTSC(algorithm="brute", n_neighbors=1, metric=this_measure, metric_params=grid.best_params_['metric_params'])
            best_model.fit(full_train_to_use,y)
            end_build_time = time.time()

            self.constituent_build_times.append(str(end_build_time-start_build_time))
            self.estimators_[dm] = best_model
            self.train_accs_by_classifier[dm] = acc
            self.train_preds_by_classifier[dm] = preds

    def predict_proba(self, X):

        # Derivative DTW (DDTW) uses the regular DTW algorithm on data that are transformed into derivatives.
        # To increase the efficiency of DDTW we can pre-transform the data into derivatives, and then call the
        # standard DTW algorithm on it, rather than transforming each series every time a distance calculation
        # is made. Please note that using DDTW elsewhere will not benefit from this speed enhancement
        if self.distance_measures.__contains__(ddtw_c) or self.distance_measures.__contains__(wddtw_c):
            der_X = DerivativeSlopeTransformer().transform(X)
            der_X = np.array([np.asarray([x]).reshape(len(x), 1) for x in der_X.iloc[:, self.dim_to_use]])
        else:
            der_X = None

        # reshape X for use with the efficient cython distance measures
        X = np.array([np.asarray([x]).reshape(len(x),1) for x in X.iloc[:, self.dim_to_use]])

        output_probas = []
        train_sum = 0

        for c in range(0, len(self.estimators_)):
            if self.distance_measures[c] == ddtw_c or self.distance_measures[c] == wddtw_c:
                test_X_to_use = der_X
            else:
                test_X_to_use = X
            this_train_acc = self.train_accs_by_classifier[c]
            this_probas = np.multiply(self.estimators_[c].predict_proba(test_X_to_use), this_train_acc)
            output_probas.append(this_probas)
            train_sum += this_train_acc

        output_probas = np.sum(output_probas,axis=0)
        output_probas = np.divide(output_probas, train_sum)
        return output_probas

    def predict(self, X, return_preds_and_probas=False):
        probas = self.predict_proba(X) # does derivative transform within (if required)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        if return_preds_and_probas is False:
            return preds
        else:
            return preds, probas

    def write_constituent_train_files(self, output_file_path, dataset_name, actual_y):

        for c in range(len(self.estimators_)):
            measure_name = self.distance_measures[c].__name__

            try:
                os.makedirs(str(output_file_path) + "/" + str(measure_name) + "/Predictions/" + str(dataset_name) + "/")
            except os.error:
                pass  # raises os.error if path already exists

            file = open(str(output_file_path)+"/"+str(measure_name)+"/Predictions/" + str(dataset_name) +
                        "/trainFold"+str(self.random_seed)+".csv", "w")

            # the first line of the output file is in the form of:
            # <classifierName>,<datasetName>,<train/test>
            file.write(str(measure_name)+"," + str(dataset_name) + ",train\n")

            # the second line of the output is free form and classifier-specific; usually this will record info
            # such as build time, paramater options used, any constituent model names for ensembles, etc.
            # file.write(str(self.estimators_[c].best_params_['metric_params'])+"\n")
            self.prop_train_in_param_finding
            file.write(str(self.estimators_[c].metric_params)+",build_time,"+str(self.constituent_build_times[c])+",prop_of_param_options," + str(self.prop_of_param_options) +
                       ",prop_train_in_param_finding," + str(self.prop_train_in_param_finding)+"\n")

            # third line is training acc
            file.write(str(self.train_accs_by_classifier[c])+"\n")

            for i in range(len(actual_y)):
                file.write(str(actual_y[i])+","+str(self.train_preds_by_classifier[c][i])+"\n")
            # preds would go here once stored as part of fit

            file.close()

    @staticmethod
    def _get_100_param_options(distance_measure, train_x=None, data_dim_to_use=0):

        def get_inclusive(min_val, max_val, num_vals):
            inc = (max_val - min_val) / (num_vals-1)
            return np.arange(min_val, max_val + inc, inc)

        if distance_measure == dtw_c or distance_measure == ddtw_c:
            return {'metric_params': [{'w': x / 100} for x in range(0, 100)]}
        elif distance_measure == wdtw_c or distance_measure == wddtw_c:
            return {'metric_params': [{'g': x / 100} for x in range(0, 100)]}
        elif distance_measure == lcss_c:
            train_std = np.std(train_x)
            epsilons = get_inclusive(train_std*.2, train_std, 10)
            deltas = get_inclusive(int(len(train_x[0])/4),len(train_x[0]), 10)
            deltas = [int(d) for d in deltas]
            a = list(product(epsilons, deltas))
            return {'metric_params': [{'epsilon': a[x][0],'delta':a[x][1]} for x in range(0, len(a))]}
        elif distance_measure == erp_c:
            train_std = np.std(train_x)
            band_sizes = get_inclusive(0, 0.25, 10)
            g_vals = get_inclusive(train_std * .2, train_std, 10)
            a = list(product(band_sizes, g_vals))
            return {'metric_params': [{'band_size': a[x][0], 'g': a[x][1]} for x in range(0, len(a))]}
        elif distance_measure == msm_c:
            a = get_inclusive(0.01, 0.1, 25)
            b = get_inclusive(0.1, 1, 26)
            c = get_inclusive(1, 10, 26)
            d = get_inclusive(10,100,26)
            return {'metric_params': [{'c': x} for x in np.concatenate([a,b[1:],c[1:],d[1:]])]}
        # elif distance_measure == twe_distance
        else:
            raise NotImplementedError("EE does not currently support: " + str(distance_measure))
