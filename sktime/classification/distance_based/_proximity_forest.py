# -*- coding: utf-8 -*-
"""Proximity Forest time series classifier.

A decision tree forest which uses distance measures to partition data.
"""

# linkedin.com/goastler; github.com/goastler
__author__ = ["George Oastler"]
__all__ = ["ProximityForest", "_CachedTransformer", "ProximityStump", "ProximityTree"]

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from sktime.distances.elastic_cython import dtw_distance
from sktime.distances.elastic_cython import erp_distance
from sktime.distances.elastic_cython import lcss_distance
from sktime.distances.elastic_cython import msm_distance
from sktime.distances.elastic_cython import twe_distance
from sktime.distances.elastic_cython import wdtw_distance
from sktime.classification.distance_based._proximity_forest_utils import max as _max
from sktime.classification.distance_based._proximity_forest_utils import (
    arg_min as _arg_min,
)
from sktime.classification.base import BaseClassifier
from sktime.classification.distance_based._proximity_forest_utils import (
    positive_dataframe_indices,
    max_instance_length,
    negative_dataframe_indices,
)
from sktime.classification.distance_based._proximity_forest_utils import stdp as _stdp
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.panel.summarize import DerivativeSlopeTransformer
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y

# todo unit tests / sort out current unit tests
# todo logging package rather than print to screen
# todo get params avoid func pointer - use name
# todo set params use func name or func pointer
# todo constructor accept str name func / pointer
# todo duck-type functions


class _CachedTransformer(_PanelToPanelTransformer):
    """Transformer container.

    Transforms data and adds the transformed version to a cache. If the
    transformation is called again on already seen data the data is
    fetched from the cache rather than performing the expensive transformation.

    Parameters
    ----------
    transformer : the transformer to transform uncached data

    Attributes
    ----------
    cache       : location to store transforms seen before for fast look up

    """

    _required_parameters = ["transformer"]

    def __init__(self, transformer):
        self.cache = {}
        self.transformer = transformer
        super(_CachedTransformer, self).__init__()

    def clear(self):
        """Clear the cache."""
        self.cache = {}

    def transform(self, X, y=None):
        """Fit transformer, creating a cache for transformation.

        Parameters
        ----------
        X : pandas DataFrame of shape [n_instances, n_features]
            Input data
        y : pandas Series, shape (n_instances), optional
            Targets for supervised learning.

        Returns
        -------
        cached_instances.
        """
        # for each instance, get transformed instance from cache or
        # transform and add to cache
        cached_instances = {}
        uncached_indices = []
        for index in X.index.values:
            try:
                cached_instances[index] = self.cache[index]
            except Exception:
                uncached_indices.append(index)
        if len(uncached_indices) > 0:
            uncached_instances = X.loc[uncached_indices, :]
            transformed_uncached_instances = self.transformer.fit_transform(
                uncached_instances
            )
            transformed_uncached_instances.index = uncached_instances.index
            transformed_uncached_instances = transformed_uncached_instances.to_dict(
                "index"
            )
            self.cache.update(transformed_uncached_instances)
            cached_instances.update(transformed_uncached_instances)
        cached_instances = pd.DataFrame.from_dict(cached_instances, orient="index")
        return cached_instances

    def __str__(self):
        """Return the transformer string."""
        return self.transformer.__str__()


def _derivative_distance(distance_measure, transformer):
    """Take derivative before conducting distance measure.

    :param distance_measure: the distance measure to use
    :param transformer: the transformer to use
    :return: a distance measure function with built in transformation
    """

    def distance(instance_a, instance_b, **params):
        df = pd.DataFrame([instance_a, instance_b])
        df = transformer.transform(X=df)
        instance_a = df.iloc[0, :]
        instance_b = df.iloc[1, :]
        return distance_measure(instance_a, instance_b, **params)

    return distance


def distance_predefined_params(distance_measure, **params):
    """Conduct distance measurement with a predefined set of parameters.

    :param distance_measure: the distance measure to use
    :param params: the parameters to use in the distance measure
    :return: a distance measure with no parameters
    """

    def distance(instance_a, instance_b):
        return distance_measure(instance_a, instance_b, **params)

    return distance


def cython_wrapper(distance_measure):
    """Wrap a distance measure in cython conversion.

     Converts to 1 column per dimension format.
    :param distance_measure: distance measure to wrap
    :return: a distance measure which automatically formats data for cython
    distance measures
    """

    def distance(instance_a, instance_b, **params):
        # find distance
        instance_a = from_nested_to_2d_array(
            instance_a, return_numpy=True
        )  # todo use specific
        # dimension rather than whole
        # thing?
        instance_b = from_nested_to_2d_array(
            instance_b, return_numpy=True
        )  # todo use specific
        # dimension rather than whole thing?
        instance_a = np.transpose(instance_a)
        instance_b = np.transpose(instance_b)
        return distance_measure(instance_a, instance_b, **params)

    return distance


def pure(y):
    """Test whether a set of class labels are pure (i.e. all the same).

    Parameters
    ----------
    y : 1d array like
        array of class labels

    Returns
    -------
    result : boolean
        whether the set of class labels is pure
    """
    # get unique class labels
    unique_class_labels = np.unique(np.array(y))
    # if more than 1 unique then not pure
    return len(unique_class_labels) <= 1


def gini_gain(y, y_subs):
    """Get gini score of a split, i.e. the gain from parent to children.

    Parameters
    ----------
    y : 1d array like
        array of class labels at parent
    y_subs : list of 1d array like
        list of array of class labels, one array per child

    Returns
    -------
    score : float
        gini score of the split from parent class labels to children. Note a
        higher score means better gain,
        i.e. a better split
    """
    y = np.array(y)
    # find number of instances overall
    parent_n_instances = y.shape[0]
    # if parent has no instances then is pure
    if parent_n_instances == 0:
        for child in y_subs:
            if len(child) > 0:
                raise ValueError("children populated but parent empty")
        return 0.5
    # find gini for parent node
    score = gini(y)
    # sum the children's gini scores
    for index in range(len(y_subs)):
        child_class_labels = y_subs[index]
        # ignore empty children
        if len(child_class_labels) > 0:
            # find gini score for this child
            child_score = gini(child_class_labels)
            # weight score by proportion of instances at child compared to
            # parent
            child_size = len(child_class_labels)
            child_score *= child_size / parent_n_instances
            # add to cumulative sum
            score -= child_score
    return score


def gini(y):
    """Get gini score at a specific node.

    Parameters
    ----------
    y : 1d numpy array
        array of class labels

    Returns
    -------
    score : float
        gini score for the set of class labels (i.e. how pure they are). A
        larger score means more impurity. Zero means
        pure.
    """
    y = np.array(y)
    # get number instances at node
    n_instances = y.shape[0]
    if n_instances > 0:
        # count each class
        unique_class_labels, class_counts = np.unique(y, return_counts=True)
        # subtract class entropy from current score for each class
        class_counts = np.divide(class_counts, n_instances)
        class_counts = np.power(class_counts, 2)
        sum = np.sum(class_counts)
        return 1 - sum
    else:
        # y is empty, therefore considered pure
        raise ValueError(" y empty")


def get_one_exemplar_per_class_proximity(proximity):
    """Unpack proximity object into X, y and random_state for picking exemplars.

    Parameters
    ----------
    proximity : Proximity object
        Proximity like object containing the X, y and random_state variables
        required for picking exemplars.

    Returns
    -------
    result : function
        function choosing one exemplar per class
    """
    return get_one_exemplar_per_class(proximity.X, proximity.y, proximity.random_state)


def get_one_exemplar_per_class(X, y, random_state):
    """Pick one exemplar instance per class in the dataset.

    Parameters
    ----------
    X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,
            the column _dim_to_use is extracted
    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        The class labels.
    random_state : numpy RandomState
        a random state for sampling random numbers

    Returns
    -------
    chosen_instances : list
        list of the chosen exemplar instances.
    chosen_class_labels : array
        list of corresponding class labels for each of the chosen exemplar
        instances.
    """
    # find unique class labels
    unique_class_labels = np.unique(y)
    n_unique_class_labels = len(unique_class_labels)
    chosen_instances = [None] * n_unique_class_labels
    # for each class randomly choose and instance
    for class_label_index in range(n_unique_class_labels):
        class_label = unique_class_labels[class_label_index]
        # filter class labels for desired class and get indices
        indices = np.argwhere(y == class_label)
        # flatten numpy output
        indices = np.ravel(indices)
        # random choice
        index = random_state.choice(indices)
        # record exemplar instance and class label
        instance = X.iloc[index, :]
        chosen_instances[class_label_index] = instance
    # convert lists to numpy arrays
    return chosen_instances, unique_class_labels


def dtw_distance_measure_getter(X):
    """Generate the dtw distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {
        "distance_measure": [cython_wrapper(dtw_distance)],
        "w": stats.uniform(0, 0.25),
    }


def msm_distance_measure_getter(X):
    """Generate the msm distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    n_dimensions = 1  # todo use other dimensions
    return {
        "distance_measure": [cython_wrapper(msm_distance)],
        "dim_to_use": stats.randint(low=0, high=n_dimensions),
        "c": [
            0.01,
            0.01375,
            0.0175,
            0.02125,
            0.025,
            0.02875,
            0.0325,
            0.03625,
            0.04,
            0.04375,
            0.0475,
            0.05125,
            0.055,
            0.05875,
            0.0625,
            0.06625,
            0.07,
            0.07375,
            0.0775,
            0.08125,
            0.085,
            0.08875,
            0.0925,
            0.09625,
            0.1,
            0.136,
            0.172,
            0.208,
            0.244,
            0.28,
            0.316,
            0.352,
            0.388,
            0.424,
            0.46,
            0.496,
            0.532,
            0.568,
            0.604,
            0.64,
            0.676,
            0.712,
            0.748,
            0.784,
            0.82,
            0.856,
            0.892,
            0.928,
            0.964,
            1,
            1.36,
            1.72,
            2.08,
            2.44,
            2.8,
            3.16,
            3.52,
            3.88,
            4.24,
            4.6,
            4.96,
            5.32,
            5.68,
            6.04,
            6.4,
            6.76,
            7.12,
            7.48,
            7.84,
            8.2,
            8.56,
            8.92,
            9.28,
            9.64,
            10,
            13.6,
            17.2,
            20.8,
            24.4,
            28,
            31.6,
            35.2,
            38.8,
            42.4,
            46,
            49.6,
            53.2,
            56.8,
            60.4,
            64,
            67.6,
            71.2,
            74.8,
            78.4,
            82,
            85.6,
            89.2,
            92.8,
            96.4,
            100,
        ],
    }


def erp_distance_measure_getter(X):
    """Generate the erp distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    stdp = _stdp(X)
    instance_length = max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    n_dimensions = 1  # todo use other dimensions
    return {
        "distance_measure": [cython_wrapper(erp_distance)],
        "dim_to_use": stats.randint(low=0, high=n_dimensions),
        "g": stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp),
        "band_size": stats.randint(low=0, high=max_raw_warping_window + 1)
        # scipy stats randint is exclusive on the max value, hence + 1
    }


def lcss_distance_measure_getter(X):
    """Generate the lcss distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    stdp = _stdp(X)
    instance_length = max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    n_dimensions = 1  # todo use other dimensions
    return {
        "distance_measure": [cython_wrapper(lcss_distance)],
        "dim_to_use": stats.randint(low=0, high=n_dimensions),
        "epsilon": stats.uniform(0.2 * stdp, stdp - 0.2 * stdp),
        # scipy stats randint is exclusive on the max value, hence + 1
        "delta": stats.randint(low=0, high=max_raw_warping_window + 1),
    }


def twe_distance_measure_getter(X):
    """Generate the twe distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {
        "distance_measure": [cython_wrapper(twe_distance)],
        "penalty": [
            0,
            0.011111111,
            0.022222222,
            0.033333333,
            0.044444444,
            0.055555556,
            0.066666667,
            0.077777778,
            0.088888889,
            0.1,
        ],
        "stiffness": [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    }


def wdtw_distance_measure_getter(X):
    """Generate the wdtw distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {
        "distance_measure": [cython_wrapper(wdtw_distance)],
        "g": stats.uniform(0, 1),
    }


def euclidean_distance_measure_getter(X):
    """Generate the ed distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {"distance_measure": [cython_wrapper(dtw_distance)], "w": [0]}


def setup_wddtw_distance_measure_getter(transformer):
    """Generate the wddtw distance measure.

    Bakes the derivative transformer into the dtw distance measure
    :param transformer: the transformer to use
    :return: a getter to produce the distance measure
    """

    def getter(X):
        return {
            "distance_measure": [
                _derivative_distance(cython_wrapper(wdtw_distance), transformer)
            ],
            "g": stats.uniform(0, 1),
        }

    return getter


def setup_ddtw_distance_measure_getter(transformer):
    """Generate the ddtw distance measure.

    Bakes the derivative transformer into the dtw distance measure
    :param transformer: the transformer to use
    :return: a getter to produce the distance measure
    """

    def getter(X):
        return {
            "distance_measure": [
                _derivative_distance(cython_wrapper(dtw_distance), transformer)
            ],
            "w": stats.uniform(0, 0.25),
        }

    return getter


def setup_all_distance_measure_getter(proximity):
    """All distance measure getter functions from a proximity object.

    :param proximity: a PT / PF / PS
    :return: a list of distance measure getters
    """
    transformer = _CachedTransformer(DerivativeSlopeTransformer())
    distance_measure_getters = [
        euclidean_distance_measure_getter,
        dtw_distance_measure_getter,
        setup_ddtw_distance_measure_getter(transformer),
        wdtw_distance_measure_getter,
        setup_wddtw_distance_measure_getter(transformer),
        msm_distance_measure_getter,
        lcss_distance_measure_getter,
        erp_distance_measure_getter,
        twe_distance_measure_getter,
    ]

    def pick_rand_distance_measure(proximity):
        """Generate a distance measure from a range of parameters.

        :param proximity: proximity object containing distance measures,
        ranges and dataset
        :return: a distance measure with no parameters
        """
        random_state = proximity.random_state
        X = proximity.X
        distance_measure_getter = random_state.choice(distance_measure_getters)
        distance_measure_perm = distance_measure_getter(X)
        param_perm = pick_rand_param_perm_from_dict(distance_measure_perm, random_state)
        distance_measure = param_perm["distance_measure"]
        del param_perm["distance_measure"]
        return distance_predefined_params(distance_measure, **param_perm)

    return pick_rand_distance_measure


def pick_rand_param_perm_from_dict(param_pool, random_state):
    """Pick a parameter permutation.

    Given a list of dictionaries contain potential values OR a list of values OR a
    distribution of values (a distribution must have the .rvs() function to sample
    values)

    Parameters
    ----------
    param_pool : list of dicts OR list OR distribution
        parameters in the same format as GridSearchCV from scikit-learn.
        example:
        param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [{'C': [1, 10, 100, 1000],
          'kernel': ['linear']}],
          'kernel': ['rbf']},
         ]

    Returns
    -------
    param_perm : dict
        distance measure and corresponding parameters in dictionary format
    """
    # construct empty permutation
    param_perm = {}
    # for each parameter
    for param_name, param_values in param_pool.items():
        # if it is a list
        if isinstance(param_values, list):
            # randomly pick a value
            param_value = param_values[random_state.randint(len(param_values))]
            # if the value is another dict then get a random parameter
            # permutation from that dict (recursive over
            # 2 funcs)
            # if isinstance(param_value, dict): # no longer require
            # recursive param perms
            #     param_value = _pick_param_permutation(param_value,
            #     random_state)
        # else if parameter is a distribution
        elif hasattr(param_values, "rvs"):
            # sample from the distribution
            param_value = param_values.rvs(random_state=random_state)
        else:
            # otherwise we don't know how to obtain a value from the parameter
            raise Exception("unknown type of parameter pool")
        # add parameter name and value to permutation
        param_perm[param_name] = param_value
    return param_perm


def pick_rand_param_perm_from_list(params, random_state):
    """Get a random parameter permutation.

    Permutation providing a distance measure and corresponding parameters.

    Parameters
    ----------
    params : list of dicts
        parameters in the same format as GridSearchCV from scikit-learn.
        example:
        param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [{'C': [1, 10, 100, 1000],
          'kernel': ['linear']}], 'kernel': ['rbf']},
         ]

    Returns
    -------
    permutation : dict
        distance measure and corresponding parameters in dictionary format
    """
    #
    param_pool = random_state.choice(params)
    permutation = pick_rand_param_perm_from_dict(param_pool, random_state)
    return permutation


def best_of_n_stumps(n):
    """Generate the function to pick the best of n stump evaluations.

    Parameters
    ----------
    n : int
        the number of stumps to evaluate before picking the best. Must be 1
        or more.

    Returns
    -------
    find_best_stump : func
        function to find the best of n stumps.
    """
    if n < 1:
        raise ValueError("n cannot be less than 1")

    def find_best_stump(proximity):
        """Pick the best of n stump evaluations.

        Parameters
        ----------
        proximity : Proximity like object
            the proximity object to split data from.

        Returns
        -------
        stump : ProximityStump
            the best stump / split of data of the n attempts.
        """
        stumps = []
        # for n stumps
        for _ in range(n):
            # duplicate tree configuration
            stump = ProximityStump(
                random_state=proximity.random_state,
                get_exemplars=proximity.get_exemplars,
                distance_measure=proximity.distance_measure,
                setup_distance_measure=proximity.setup_distance_measure,
                get_distance_measure=proximity.get_distance_measure,
                get_gain=proximity.get_gain,
                verbosity=proximity.verbosity,
                n_jobs=proximity.n_jobs,
            )
            # grow the stump
            stump.fit(proximity.X, proximity.y)
            stump.grow()
            stumps.append(stump)
        # pick the best stump based upon gain
        stump = _max(stumps, proximity.random_state, lambda stump: stump.entropy)
        return stump

    return find_best_stump


class ProximityStump(BaseClassifier):
    """Proximity Stump class.

    Model a decision stump which uses a distance measure to partition data.

    Attributes
    ----------
        label_encoder: label encoder to change string labels to numeric indices
        y_exemplar: class label list of the exemplar instances
        X_exemplar: dataframe of the exemplar instances
        X_branches: dataframes for each branch, one per exemplar
        y_branches: class label list for each branch, one per exemplar
        classes_: unique list of classes
        entropy: the gain associated with the split of data
        random_state: the random state
        get_exemplars: function to extract exemplars from a dataframe and
        class value list
        setup_distance_measure: function to setup the distance measure
        getters from dataframe and class value list
        get_distance_measure: distance measure getters
        distance_measure: distance measures
        get_gain: function to score the quality of a split
        verbosity: logging verbosity
        n_jobs: number of jobs to run in parallel *across threads"
    """

    __author__ = "George Oastler (linkedin.com/goastler; github.com/goastler)"

    def __init__(
        self,
        random_state=None,
        get_exemplars=get_one_exemplar_per_class_proximity,
        setup_distance_measure=setup_all_distance_measure_getter,
        get_distance_measure=None,
        distance_measure=None,
        get_gain=gini_gain,
        verbosity=0,
        n_jobs=1,
    ):
        """
        Construct a proximity stump.

        :param random_state: the random state
        :param get_exemplars: function to extract exemplars from a dataframe
        and class value list
        :param setup_distance_measure: function to setup the distance
        measure getters from dataframe and class value list
        :param get_distance_measure: distance measure getters
        :param distance_measure: distance measures
        :param get_gain: function to score the quality of a split
        :param verbosity: logging verbosity
        :param n_jobs: number of jobs to run in parallel *across threads"
        """
        self.setup_distance_measure = setup_distance_measure
        self.random_state = random_state
        self.get_distance_measure = get_distance_measure
        self.distance_measure = distance_measure
        self.pick_exemplars = get_exemplars
        self.get_gain = get_gain
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        # set in fit
        self.label_encoder = None
        self.y_exemplar = None
        self.X_exemplar = None
        self.X_branches = None
        self.y_branches = None
        self.X = None
        self.y = None
        self.classes_ = None
        self.entropy = None
        super(ProximityStump, self).__init__()

    @staticmethod
    def _distance_to_exemplars_inst(exemplars, instance, distance_measure):
        """Find distance between a given instance and the exemplar instances.

        :param exemplars: the exemplars to use
        :param instance: the instance to compare to each exemplar
        :param distance_measure: the distance measure to provide similarity
        values
        :return: list of distances to each exemplar
        """
        n_exemplars = len(exemplars)
        distances = np.empty(n_exemplars)
        min_distance = np.math.inf
        for exemplar_index in range(n_exemplars):
            exemplar = exemplars[exemplar_index]
            if exemplar.name == instance.name:
                distance = 0
            else:
                distance = distance_measure(instance, exemplar)  # , min_distance)
            if distance < min_distance:
                min_distance = distance
            distances[exemplar_index] = distance
        return distances

    def distance_to_exemplars(self, X):
        """Find distance to exemplars.

        Parameters
        ----------
        X: the dataset containing a list of instances

        Return
        ------
        2d numpy array of distances from each instance to each
        exemplar (instance by exemplar)
        """
        check_X(X)
        if self.n_jobs > 1 or self.n_jobs < 0:
            parallel = Parallel(self.n_jobs)
            distances = parallel(
                delayed(self._distance_to_exemplars_inst)(
                    self.X_exemplar, X.iloc[index, :], self.distance_measure
                )
                for index in range(X.shape[0])
            )
        else:
            distances = [
                self._distance_to_exemplars_inst(
                    self.X_exemplar, X.iloc[index, :], self.distance_measure
                )
                for index in range(X.shape[0])
            ]
        distances = np.vstack(np.array(distances))
        return distances

    def fit(self, X, y):
        """
        Build the classifier on the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,
            column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_pandas=True)

        self.X = positive_dataframe_indices(X)
        self.random_state = check_random_state(self.random_state)
        # setup label encoding
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        self.y = y
        self.classes_ = self.label_encoder.classes_
        if self.distance_measure is None:
            if self.get_distance_measure is None:
                self.get_distance_measure = self.setup_distance_measure(self)
            self.distance_measure = self.get_distance_measure(self)
        self.X_exemplar, self.y_exemplar = self.pick_exemplars(self)
        self._is_fitted = True
        return self

    def find_closest_exemplar_indices(self, X):
        """Find the closest exemplar index for each instance in a dataframe.

        Parameters
        ----------
        X: the dataframe containing instances

        Return
        ------
        1d numpy array of indices, one for each instance,
        reflecting the index of the closest exemplar
        """
        check_X(X)  # todo make checks optional and propogate from forest downwards
        n_instances = X.shape[0]
        distances = self.distance_to_exemplars(X)
        indices = np.empty(X.shape[0], dtype=int)
        for index in range(n_instances):
            exemplar_distances = distances[index]
            closest_exemplar_index = _arg_min(exemplar_distances, self.random_state)
            indices[index] = closest_exemplar_index
        return indices

    def grow(self):
        """Grow the stump, creating branches for each exemplar."""
        n_exemplars = len(self.y_exemplar)
        indices = self.find_closest_exemplar_indices(self.X)
        self.X_branches = [None] * n_exemplars
        self.y_branches = [None] * n_exemplars
        for index in range(n_exemplars):
            instance_indices = np.argwhere(indices == index)
            instance_indices = np.ravel(instance_indices)
            self.X_branches[index] = self.X.iloc[instance_indices, :]
            y = np.take(self.y, instance_indices)
            self.y_branches[index] = y
        self.entropy = self.get_gain(self.y, self.y_branches)
        return self

    def predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it
            only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        X = check_X(X, enforce_univariate=True, coerce_to_pandas=True)

        X = negative_dataframe_indices(X)
        distances = self.distance_to_exemplars(X)
        ones = np.ones(distances.shape)
        distances = np.add(distances, ones)
        distributions = np.divide(ones, distances)
        normalize(distributions, copy=False, norm="l1")
        return distributions


class ProximityTree(BaseClassifier):
    """Proximity Tree class.

    A decision tree which uses distance measures to partition data.

    Attributes
    ----------
        label_encoder: label encoder to change string labels to numeric indices
        classes_: unique list of classes
        random_state: the random state
        get_exemplars: function to extract exemplars from a dataframe and
        class value list
        setup_distance_measure: function to setup the distance measure
        getters from dataframe and class value list
        get_distance_measure: distance measure getters
        distance_measure: distance measures
        get_gain: function to score the quality of a split
        verbosity: logging verbosity
        n_jobs: number of jobs to run in parallel *across threads"
        find_stump: function to find the best split of data
        max_depth: max tree depth
        depth: current depth of tree, as each node is a tree itself,
        therefore can have a depth of >=0
        X: train data
        y: train data labels
        stump: the stump used to split data at this node
        branches: the partitions of data driven by the stump
    """

    def __init__(
        self,
        # note: any changes of these params must be reflected in
        # the fit method for building trees / clones
        random_state=None,
        get_exemplars=get_one_exemplar_per_class_proximity,
        distance_measure=None,
        get_distance_measure=None,
        setup_distance_measure=setup_all_distance_measure_getter,
        get_gain=gini_gain,
        max_depth=np.math.inf,
        is_leaf=pure,
        verbosity=0,
        n_jobs=1,
        n_stump_evaluations=5,
        find_stump=None,
    ):
        """Build a Proximity Tree object.

        :param random_state: the random state
        :param get_exemplars: get the exemplars from a given dataframe and
        list of class labels
        :param distance_measure: distance measure to use
        :param get_distance_measure: method to get the distance measure if
        no already set
        :param setup_distance_measure: method to setup the distance measures
        based upon the dataset given
        :param get_gain: method to find the gain of a data split
        :param max_depth: maximum depth of the tree
        :param is_leaf: function to decide when to mark a node as a leaf node
        :param verbosity: number reflecting the verbosity of logging
        :param n_jobs: number of parallel threads to use while building
        :param find_stump: method to find the best split of data / stump at
        a node
        :param n_stump_evaluations: number of stump evaluations to do if
        find_stump method is None
        """
        self.verbosity = verbosity
        self.n_stump_evaluations = n_stump_evaluations
        self.find_stump = find_stump
        self.max_depth = max_depth
        self.get_distance_measure = distance_measure
        self.random_state = random_state
        self.is_leaf = is_leaf
        self.get_distance_measure = get_distance_measure
        self.setup_distance_measure = setup_distance_measure
        self.get_exemplars = get_exemplars
        self.get_gain = get_gain
        self.n_jobs = n_jobs
        self.depth = 0
        # below set in fit method
        self.label_encoder = None
        self.distance_measure = None
        self.stump = None
        self.branches = None
        self.X = None
        self.y = None
        self.classes_ = None
        super(ProximityTree, self).__init__()

    def fit(self, X, y):
        """Build the classifier on the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,
            column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_pandas=True)
        self.X = positive_dataframe_indices(X)
        self.random_state = check_random_state(self.random_state)
        if self.find_stump is None:
            self.find_stump = best_of_n_stumps(self.n_stump_evaluations)
        # setup label encoding
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        self.y = y
        self.classes_ = self.label_encoder.classes_
        if self.distance_measure is None:
            if self.get_distance_measure is None:
                self.get_distance_measure = self.setup_distance_measure(self)
            self.distance_measure = self.get_distance_measure(self)
        self.stump = self.find_stump(self)
        n_branches = len(self.stump.y_exemplar)
        self.branches = [None] * n_branches
        if self.depth < self.max_depth:
            for index in range(n_branches):
                sub_y = self.stump.y_branches[index]
                if not self.is_leaf(sub_y):
                    sub_tree = ProximityTree(
                        random_state=self.random_state,
                        get_exemplars=self.get_exemplars,
                        distance_measure=self.distance_measure,
                        setup_distance_measure=self.setup_distance_measure,
                        get_distance_measure=self.get_distance_measure,
                        get_gain=self.get_gain,
                        is_leaf=self.is_leaf,
                        verbosity=self.verbosity,
                        max_depth=self.max_depth,
                        n_jobs=self.n_jobs,
                    )
                    sub_tree.label_encoder = self.label_encoder
                    sub_tree.depth = self.depth + 1
                    self.branches[index] = sub_tree
                    sub_X = self.stump.X_branches[index]
                    sub_tree.fit(sub_X, sub_y)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it
            only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        X = check_X(X, enforce_univariate=True, coerce_to_pandas=True)
        X = negative_dataframe_indices(X)
        closest_exemplar_indices = self.stump.find_closest_exemplar_indices(X)
        n_classes = len(self.label_encoder.classes_)
        distribution = np.zeros((X.shape[0], n_classes))
        for index in range(len(self.branches)):
            indices = np.argwhere(closest_exemplar_indices == index)
            if indices.shape[0] > 0:
                indices = np.ravel(indices)
                sub_tree = self.branches[index]
                if sub_tree is None:
                    sub_distribution = np.zeros((1, n_classes))
                    class_label = self.stump.y_exemplar[index]
                    sub_distribution[0][class_label] = 1
                else:
                    sub_X = X.iloc[indices, :]
                    sub_distribution = sub_tree.predict_proba(sub_X)
                assert sub_distribution.shape[1] == n_classes
                np.add.at(distribution, indices, sub_distribution)
        normalize(distribution, copy=False, norm="l1")
        return distribution


class ProximityForest(BaseClassifier):
    """Proximity Forest class.

    Models a decision tree forest which uses distance measures to partition data [1].

    Parameters
    ----------
    random_state: random, default = None
        seed for reproducibility
    n_estimators : int, default=100
        The number of trees in the forest.
    distance_measure: default = None
    get_distance_measure: default=None,
        distance measure getters
    get_exemplars: default=get_one_exemplar_per_class_proximity,
    get_gain: default=gini_gain,
            function to score the quality of a split
    verbosity: default=0,
            logging verbosity
    max_depth: default=np.math.inf,
    is_leaf: default=pure,
    n_jobs: default=int, 1,
        number of jobs to run in parallel *across threads"
    n_stump_evaluations: int, default=5,
    find_stump: default=None,
        function to find the best split of data
    setup_distance_measure_getter=setup_all_distance_measure_getter,
    setup_distance_measure_getter: function to setup the distance

    Attributes
    ----------
    label_encoder: label encoder to change string labels to numeric indices
    classes_: unique list of classes
    get_exemplars: function to extract exemplars from a dataframe and
           class value list
    max_depth: max tree depth
    X: train data
    y: train data labels
    trees: list of trees in the forest

    Notes
    -----
    ..[1] Ben Lucas et al., "Proximity Forest: an effective and scalable distance-based
      classifier for time series",Data Mining and Knowledge Discovery, 33(3): 607-635,
      2019 https://arxiv.org/abs/1808.10594
    Java wrapper of authors original
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/distance_based/ProximityForestWrapper.java
    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/distance_based/proximity/ProximityForest.java

    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        random_state=None,
        n_estimators=100,
        distance_measure=None,
        get_distance_measure=None,
        get_exemplars=get_one_exemplar_per_class_proximity,
        get_gain=gini_gain,
        verbosity=0,
        max_depth=np.math.inf,
        is_leaf=pure,
        n_jobs=1,
        n_stump_evaluations=5,
        find_stump=None,
        setup_distance_measure_getter=setup_all_distance_measure_getter,
    ):
        """Build a Proximity Forest object.

        Parameters
        ----------
        random_state: the random state
        get_exemplars: get the exemplars from a given dataframe and
        list of class labels
        distance_measure: distance measure to use
        get_distance_measure: method to get the distance measure if
        no already set
        setup_distance_measure_getter: method to setup the distance
        measures based upon the dataset given
        get_gain: method to find the gain of a data split
        :param max_depth: maximum depth of the tree
        :param is_leaf: function to decide when to mark a node as a leaf node
        :param verbosity: number reflecting the verbosity of logging
        :param n_jobs: number of parallel threads to use while building
        :param find_stump: method to find the best split of data / stump at
        a node
        :param n_stump_evaluations: number of stump evaluations to do if
        find_stump method is None
        :param n_estimators: number of trees to construct
        """
        self.is_leaf = is_leaf
        self.verbosity = verbosity
        self.max_depth = max_depth
        self.get_exemplars = get_exemplars
        self.get_gain = get_gain
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.n_stump_evaluations = n_stump_evaluations
        self.get_distance_measure = get_distance_measure
        self.setup_distance_measure_getter = setup_distance_measure_getter
        self.distance_measure = distance_measure
        self.find_stump = find_stump
        # set in fit method
        self.label_encoder = None
        self.trees = None
        self.X = None
        self.y = None
        self.classes_ = None
        super(ProximityForest, self).__init__()

    def _fit_tree(self, X, y, index, random_state):
        """Build the classifierr on the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed,
            column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.
        index : index of the tree to be constructed
        random_state: random_state to send to the tree to be constructed

        Returns
        -------
        self : object
        """
        if self.verbosity > 0:
            print("tree " + str(index) + " building")  # noqa
        tree = ProximityTree(
            random_state=random_state,
            verbosity=self.verbosity,
            get_exemplars=self.get_exemplars,
            get_gain=self.get_gain,
            distance_measure=self.distance_measure,
            setup_distance_measure=self.setup_distance_measure_getter,
            get_distance_measure=self.get_distance_measure,
            max_depth=self.max_depth,
            is_leaf=self.is_leaf,
            n_jobs=1,
            find_stump=self.find_stump,
            n_stump_evaluations=self.n_stump_evaluations,
        )
        tree.fit(X, y)
        return tree

    def fit(self, X, y):
        """Build the classifier on the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,
            column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_pandas=True)
        self.X = positive_dataframe_indices(X)
        self.random_state = check_random_state(self.random_state)
        # setup label encoding
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        self.y = y
        self.classes_ = self.label_encoder.classes_
        if self.distance_measure is None:
            if self.get_distance_measure is None:
                self.get_distance_measure = self.setup_distance_measure_getter(self)
            self.distance_measure = self.get_distance_measure(self)
        if self.n_jobs > 1 or self.n_jobs < 0:
            parallel = Parallel(self.n_jobs)
            self.trees = parallel(
                delayed(self._fit_tree)(
                    X, y, index, self.random_state.randint(0, self.n_estimators)
                )
                for index in range(self.n_estimators)
            )
        else:
            self.trees = [
                self._fit_tree(
                    X, y, index, self.random_state.randint(0, self.n_estimators)
                )
                for index in range(self.n_estimators)
            ]
        self._is_fitted = True
        return self

    @staticmethod
    def _predict_proba_tree(X, tree):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it
            only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.
        tree : the tree to collect predictions from

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        return tree.predict_proba(X)

    def predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it
            only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        X = check_X(X, enforce_univariate=True, coerce_to_pandas=True)
        X = negative_dataframe_indices(X)
        if self.n_jobs > 1 or self.n_jobs < 0:
            parallel = Parallel(self.n_jobs)
            distributions = parallel(
                delayed(self._predict_proba_tree)(X, tree) for tree in self.trees
            )
        else:
            distributions = [self._predict_proba_tree(X, tree) for tree in self.trees]
        distributions = np.array(distributions)
        distributions = np.sum(distributions, axis=0)
        normalize(distributions, copy=False, norm="l1")
        return distributions
