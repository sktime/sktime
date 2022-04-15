# -*- coding: utf-8 -*-
"""Proximity Forest time series classifier.

A decision tree forest which uses distance measures to partition data.
B. Lucas and A. Shifaz, C. Pelletier, L. O’Neill, N. Zaidi, B. Goethals,
F. Petitjean and G. Webb
Proximity Forest: an effective and scalable distance-based classifier for
time series,
Data Mining and Knowledge Discovery, 33(3): 607-635, 2019
"""

# linkedin.com/goastler; github.com/goastler
# github.com/moradisten
__author__ = ["gastler", "Morad A. Azaz"]
__all__ = ["ProximityForest", "_CachedTransformer", "ProximityStump", "ProximityTree"]

from logging import exception

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.datatypes._panel._convert import (
    from_nested_to_2d_array,
    from_nested_to_3d_numpy,
)
from sktime.distances import (  # twe_distance,
    dtw_distance,
    erp_distance,
    lcss_distance,
    msm_distance,
    wdtw_distance,
)
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.panel.summarize import DerivativeSlopeTransformer
from sktime.utils.validation.panel import check_X, check_X_y

# todo unit tests / sort out current unit tests
# todo logging package rather than print to screen
# todo get params avoid func pointer - use name
# todo set params use func name or func pointer
# todo constructor accept str name func / pointer
# todo duck-type functions


def stdp(X):
    """Proximity forest util function (deprecated)."""
    warn(
        "This function has moved to classification/distance_based/_proximity_forest as "
        "a private function. This version will be removed in V0.10",
        FutureWarning,
    )
    sum = 0
    sum_sq = 0
    num_instances = X.shape[0]
    num_dimensions = X.shape[1]
    num_values = 0
    for instance_index in range(0, num_instances):
        for dimension_index in range(0, num_dimensions):
            instance = X.iloc[instance_index, dimension_index]
            for value in instance:
                num_values += 1
                sum += value
                sum_sq += value**2  # todo missing values NaN messes
                # this up!
    mean = sum / num_values
    stdp = np.math.sqrt(sum_sq / num_values - mean**2)
    return stdp


# find index of min value in array, randomly breaking ties
def _arg_min(array, rand, getter=None):
    """Proximity forest util function."""
    return rand.choice(arg_mins(array, getter))


def max_instance_length(X):
    """Proximity forest util function."""
    max_length = len(X.iloc[0, 0])
    # max = -1
    # for dimension in range(0, instances.shape[1]):
    #     length = max_instance_dimension_length(instances, dimension)
    #     if length > max:
    #         max = length
    return max_length


class _CachedTransformer(_PanelToPanelTransformer):
    """Transformer that transforms data and adds the transformed version to a cache.

    If the transformation is called again on already seen data the data is
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
        """
        Fit transformer, creating a cache for transformation.

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
        """Transform string."""
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


def numba_wrapper(distance_measure):
    """Wrap a distance measure in cython conversion.

    (to 1 column per dimension format)
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
        class_counts_sum = np.sum(class_counts)
        return 1 - class_counts_sum
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
        "distance_measure": [numba_wrapper(dtw_distance)],
        "w": stats.uniform(0, 0.25),
    }


def msm_distance_measure_getter(X):
    """Generate the msm distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    n_dimensions = 1  # todo use other dimensions
    return {
        "distance_measure": [numba_wrapper(dtw_distance)],
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
        "distance_measure": [numba_wrapper(erp_distance)],
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
        "distance_measure": [numba_wrapper(lcss_distance)],
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
        "distance_measure": [numba_wrapper(twe_distance)],
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
        "distance_measure": [numba_wrapper(weighted_dtw_distance)],
        "g": stats.uniform(0, 1),
    }


def euclidean_distance_measure_getter(X):
    """Generate the ed distance measure.

    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {"distance_measure": [numba_wrapper(dtw_distance)], "w": [0]}


def setup_wddtw_distance_measure_getter(transformer):
    """Generate the wddtw distance measure by baking the derivative transformer.

    into the wdtw distance measure
    :param transformer: the transformer to use
    :return: a getter to produce the distance measure
    """

    def getter(X):
        return {
            "distance_measure": [
                _derivative_distance(numba_wrapper(weighted_dtw_distance), transformer)
            ],
            "g": stats.uniform(0, 1),
        }

    return getter


def setup_ddtw_distance_measure_getter(transformer):
    """Generate the ddtw distance measure by baking the derivative transformer.

    into the dtw distance measure
    :param transformer: the transformer to use
    :return: a getter to produce the distance measure
    """

    def getter(X):
        return {
            "distance_measure": [
                _derivative_distance(numba_wrapper(dtw_distance), transformer)
            ],
            "w": stats.uniform(0, 0.25),
        }

    return getter


def setup_all_distance_measure_getter(proximity):
    """Set all distance measure getter functions from a proximity object.

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
        #        twe_distance_measure_getter,
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
    distribution of values (a distribution must have the .rvs() function to
    sample values)

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

     Providing a distance measure and corresponding parameters

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
    """Proximity Stump."""

    np.random.seed(1234)

    def __init__(
        self,
        random_state=0,
        setup_distance_measure=setup_all_distance_measure_getter,
        get_distance_measure=None,
        distance_measure=dtw_distance,
        X=None,
        y=None,
        label=None,
        verbosity=0,
        n_stumps=5,
        n_jobs=1,
    ):
        self.setup_distance_measure = setup_distance_measure
        self.random_state = random_state
        self.n_stumps = n_stumps
        self.get_distance_measure = get_distance_measure
        self.distance_measure = distance_measure
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        # set in fit
        self.num_children = None
        self.label_encoder = None
        # exemplars
        self.y_exemplar = None
        self.X_exemplar = None
        # temp_exemplars
        self.temp_exemplar = dict()
        # best_splits
        self.X_best_splits = None
        self.y_best_splits = None
        # Datasets
        self.X = X
        self.y = y
        # splits
        self.children = list()
        self.is_leaf = False
        self.classes_ = dict()
        self.label = label
        self.entropy = None
        super(ProximityStump, self).__init__()

    def set_X(self, X):
        """Set X."""
        self.X = X

    def set_y(self, y):
        """Set y."""
        self.y = y

    @staticmethod
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
        score = ProximityStump.gini(y)
        # sum the children's gini scores
        for index in range(len(y_subs)):
            child_class_labels = y_subs[index]
            # ignore empty children
            if len(child_class_labels) > 0:
                # find gini score for this child
                child_score = ProximityStump.gini(child_class_labels)
                # weight score by proportion of instances at child compared to
                # parent
                child_size = len(child_class_labels)
                child_score *= child_size / parent_n_instances
                # add to cumulative sum
                score -= child_score
        return score

    @staticmethod
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
        try:
            n_instances = y.shape[0]
        except exception:
            n_instances = 0

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
            return 0
            # raise ValueError(' y empty')

    @staticmethod
    def split_X_per_class(X, y):
        """Split by class.

        :param X: Array-like containing instances
        :param y: Array-like containing class labels
        :return: Returns a dictionary {Label: [sub_X]} in which sub_X contains all
        instances that match that label
        """
        split_class_x = dict()
        y_size = len(y)
        for index in range(y_size):
            label = y[index]
            if not split_class_x.keys().__contains__(label):
                split_class_x[label] = list()
            if X.shape == 3:
                split_class_x[label].append(X[index][0])
            else:
                split_class_x[label].append(X[index])
        return split_class_x

    @staticmethod
    def calculate_dist_to_exemplars_inst(exemplars, instance, distance_measure):
        """Calculate distance to exemplars."""
        distances = list()
        indices = list()
        if len(exemplars) == 0:
            return None
        for index in range(len(exemplars)):
            exemplar = exemplars[index]
            try:
                distance = distance_measure(instance, exemplar)
            except exception:
                distance = np.inf
            distances.append(distance)
            indices.append(index)
        return distances, indices

    @staticmethod
    def find_closest_distances_inst(exemplars, instance, distance_measure):
        """Find closest distance instance."""
        distances = list()
        indices = list()
        min_distance = np.math.inf
        if (exemplars is None) or len(exemplars) == 0:
            return None, None
        for index in range(len(exemplars)):
            exemplar = exemplars[index][0]
            try:
                distance = distance_measure(instance, exemplar)
            except Exception:
                distance = np.inf
            if len(indices) == 0:
                min_distance = distance
                distances.append(distance)
                indices.append(index)
            else:
                if distance < min_distance:
                    min_distance = distance
                    distances.clear()
                    distances.append(distance)
                    indices.clear()
                    indices.append(index)
                elif distance == min_distance:
                    distances.append(distance)
                    indices.append(index)
        return distances, indices

    @staticmethod
    def find_closest_distance(exemplars, instance, distance_measure):
        """Find closest distance."""
        distance, indices = ProximityStump.find_closest_distances_inst(
            exemplars, instance, distance_measure
        )
        if distance is None:
            return -1, -1
        elif len(distance) == 1:
            return distance[0], indices[0]
        else:
            r = np.random.randint(0, len(distance))
        return distance[r], indices[r]

    def find_closest_distance_(self, instance, distance_measure):
        """Find closest distance."""
        return ProximityStump.find_closest_distance(
            self.X_exemplar, instance, distance_measure
        )

    def find_closest_exemplar_indices(self, X):
        """Find closes exemplar indices."""
        check_X(X)  # todo make checks optional and propogate from forest downwards
        n_instances = X.shape[0]
        distances = self.distance_to_exemplars(X)
        indices = np.empty(X.shape[0], dtype=int)
        for index in range(n_instances):
            exemplar_distances = distances[index]
            closest_exemplar_index = arg_mins(exemplar_distances, self.random_state)
            indices[index] = closest_exemplar_index[0]
        return indices

    def split_stump(self, X, y, dataset_per_class):
        """Split stump."""
        splits_x = dict()  # {index: x_list}
        splits_y = dict()  # {index: y_list}
        label_branch = 0
        for label in dataset_per_class.keys():
            if len(dataset_per_class[label]) == 0:
                continue
            else:
                sub_X = dataset_per_class[
                    label
                ]  # sub_X is a list of series/arrays who belong to a label
                r = np.random.randint(0, len(sub_X))  # select a random element in sub_X
                splits_x[label_branch] = list()
                splits_y[label_branch] = list()
                self.temp_exemplar[label_branch] = sub_X[r]  # sub_X[r] is a serie
                self.classes_[label_branch] = label
                label_branch = label_branch + 1
        for j in range(X.shape[0]):
            instance = self.X[j][0]
            closest_distance, index = ProximityStump.find_closest_distance(
                self.temp_exemplar, instance, self.distance_measure
            )
            if closest_distance == -1:
                return splits_x, splits_y
            splits_x[index].append(X[j][0])
            splits_y[index].append(y[j])
        return splits_x, splits_y  # <index, list_x>, <index, list_y>

    def find_best_stumps(self, X, y):
        """Find best stumps."""
        x_per_label = self.split_X_per_class(X, y)
        best_weighted_gini = np.inf
        x_size = len(X)
        for _ in range(self.n_stumps):
            splits_x, splits_y = self.split_stump(X, y, x_per_label)
            if len(splits_x) == 0:
                return self.X_best_splits, self.y_best_splits
            weighted_gini = self.weighted_gini(x_size, splits_x, splits_y)
            if weighted_gini < best_weighted_gini:
                best_weighted_gini = weighted_gini
                self.X_best_splits = splits_x
                self.y_best_splits = splits_y
                self.X_exemplar = self.temp_exemplar
                self.y_exemplar = self.temp_exemplar.keys()
        self.num_children = len(self.X_best_splits)
        return self.X_best_splits, self.y_best_splits

    @staticmethod
    def weighted_gini(x_size, splits_x, splits_y):
        """Find weighted Gini."""
        wgini = 0.0
        for index in range(len(splits_x)):
            spt_x = splits_x[index]
            spt_y = splits_y[index]
            wgini = wgini + (len(spt_x) / x_size) * ProximityStump.gini(spt_y)
        return wgini

    def calculate_distance_to_exemplars(self, X):
        """Find distance to exemplars.

        :param X: the dataset containing a list of instances
        :return: 2d numpy array of distances from each instance to each
        exemplar (instance by exemplar)
        """
        check_X(X)
        if self.n_jobs > 1 or self.n_jobs < 0:
            parallel = Parallel(self.n_jobs)
            distances = parallel(
                delayed(self.calculate_dist_to_exemplars_inst)(
                    self.X_exemplar, X[0][index, :], self.distance_measure
                )
                for index in range(X.shape[0])
            )
        else:
            distances = [
                self.calculate_dist_to_exemplars_inst(
                    self.X_exemplar, X[index, :][0], self.distance_measure
                )
                for index in range(X.shape[0])
            ]
        return distances

    def distance_to_exemplars(self, X):
        """Distance to exemplars."""
        distances = self.calculate_distance_to_exemplars(X)
        distances = [x[0][0] for x in distances]
        distances = np.vstack(np.array(distances))
        return distances

    def distance_to_exemplars_indices(self, X):
        """Distance to exemplars indices."""
        distances_indices = self.calculate_distance_to_exemplars(X)
        indices = [x[1][0] for x in distances_indices]
        return indices

    def fit(self, X=None, y=None):
        """Fit."""
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if len(y) == 0:
            return
        gini = ProximityStump.gini(y)
        if gini == 0:
            self.label = int(y[0])
            self.is_leaf = True
            return
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        self.find_best_stumps(X, y)
        if len(self.X_best_splits) > 0:
            for i in range(0, len(self.X_best_splits.values())):
                self.children.append(
                    ProximityStump(
                        X=X, y=y, label=y[i], distance_measure=self.distance_measure
                    )
                )
            counter = 0
            for index in self.X_best_splits.keys():
                x_branches = self.X_best_splits[index]
                y_branches = self.y_best_splits[index]
                try:
                    splits_x = np.array(x_branches)
                    splits_y = np.array(y_branches)
                    if splits_x.shape == 2:
                        splits_x = splits_x.reshape(
                            (splits_x.shape[0], 1, splits_x.shape[1])
                        )
                    self.children[counter].fit(splits_x, splits_y)
                except RecursionError:
                    return
                counter = counter + 1

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
        X = check_X(X, enforce_univariate=True)
        distances = self.distance_to_exemplars(X)
        ones = np.ones(distances.shape)
        distances = np.add(distances, ones)
        distributions = np.divide(ones, distances)
        normalize(distributions, copy=False, norm="l1")
        return distributions


class ProximityTree(BaseClassifier):
    """Proximity Tree class to model a distance based decision tree.

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
        random_state=0,
        get_exemplars=get_one_exemplar_per_class_proximity,
        distance_measure=dtw_distance,
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
        super().__init__()
        self.verbosity = verbosity
        self.n_stump_evaluations = n_stump_evaluations
        self.find_stump = find_stump
        self.max_depth = max_depth
        self.get_distance_measure = distance_measure
        self.random_state = check_random_state(random_state)
        self.get_distance_measure = get_distance_measure
        self.setup_distance_measure = setup_all_distance_measure_getter
        self.get_gain = get_gain
        self.n_jobs = n_jobs
        self.depth = 0
        # below set in fit method
        self.label_encoder = None
        self.distance_measure = distance_measure
        self.root_stump = None
        self.branches = None
        self.X = None
        self.y = None
        self._is_fitted = False
        self.classes_ = None

    def fit(self, X, y, random_state=0):
        """Fit."""
        self.classes_ = np.unique(y)
        self.root_stump = ProximityStump(
            X=X,
            y=y,
            n_stumps=self.n_stump_evaluations,
            distance_measure=self.distance_measure,
            random_state=self.random_state,
        )
        self.root_stump.fit()
        self._is_fitted = True

    def predict_class_label(self, query):
        """Predict the label of a query.

        :param query:
        :return:
        """
        stump = self.root_stump
        while not stump.is_leaf:
            child_index = stump.find_closest_distance_(query, self.distance_measure)[1]
            if child_index == -1:
                stump.is_leaf = True
                continue
            stump = stump.children[child_index]
        return stump.label

    def predict_proba(self, X):
        """Predict proba."""
        n_instances = X.shape[0]
        predictions = np.zeros(n_instances, dtype=int)
        for index in range(n_instances):
            query = X[index][0]
            class_label = self.predict_class_label(query)
            predictions[index] = class_label
        return predictions


class ProximityForest(BaseClassifier):
    """Proximity Forest class to model a decision tree forest.

    Which uses distance measures to partition data.

    @article{lucas19proximity,

        title={Proximity Forest: an effective and scalable distance-based
        classifier for time series},
        author={B. Lucas and A. Shifaz and C. Pelletier and L. O’Neill and N.
        Zaidi and B. Goethals and F. Petitjean and G. Webb},
         journal={Data Mining and Knowledge Discovery},
        volume={33},
        number={3},
        pages={607--635},
        year={2019}
        }

    Attributes
    ----------
         label_encoder: label encoder to change string labels to numeric indices
         classes_: unique list of classes
         random_state: the random state
         get_exemplars: function to extract exemplars from a dataframe and
         class value list
         setup_distance_measure_getter: function to setup the distance
         measure getters from dataframe and class value list
         get_distance_measure: distance measure getters
         distance_measure: distance measures
         get_gain: function to score the quality of a split
         verbosity: logging verbosity
         n_jobs: number of jobs to run in parallel *across threads"
         find_stump: function to find the best split of data
         max_depth: max tree depth
         X: train data
         y: train data labels
         trees: list of trees in the forest
    """

    def __init__(
        self,
        random_state=0,
        n_estimators=100,
        distance_measure=dtw_distance,
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

        :param random_state: the random state
        :param get_exemplars: get the exemplars from a given dataframe and
        list of class labels
        :param distance_measure: distance measure to use
        :param get_distance_measure: method to get the distance measure if
        no already set
        :param setup_distance_measure_getter: method to setup the distance
        measures based upon the dataset given
        :param get_gain: method to find the gain of a data split
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
        self.X = None
        self.y = None
        self.classes_ = None
        self.trees = list()
        self.num_classes_predicted = dict()

        for _ in range(self.n_estimators):
            self.trees.append(
                ProximityTree(
                    n_stump_evaluations=self.n_stump_evaluations,
                    distance_measure=self.distance_measure,
                )
            )
        super(ProximityForest, self).__init__()

    def _fit_tree(self, X, y, index, random_state=0):
        self.trees[index].fit(X, y, random_state)
        return self.trees[index]

    @staticmethod
    def _predict_proba_tree(X, tree):
        return tree.predict_proba(X)

    def fit_tree(self, X, y, index, random_state):
        """Build the classifier on the training set (X, y).

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
            distance_measure=self.distance_measure,
            get_distance_measure=self.get_distance_measure,
            max_depth=self.max_depth,
            n_jobs=1,
            find_stump=self.find_stump,
            n_stump_evaluations=self.n_stump_evaluations,
        )
        tree.fit(X, y)
        return tree

    def fit(self, X, y):
        """Fit."""
        X, y = check_X_y(X, y, enforce_univariate=True)
        X = from_nested_to_3d_numpy(X)
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
        X = from_nested_to_3d_numpy(X)
        X = check_X(X, enforce_univariate=True)
        if self.n_jobs > 1 or self.n_jobs < 0:
            parallel = Parallel(self.n_jobs)
            predictions_per_tree = parallel(
                delayed(self._predict_proba_tree)(X, tree) for tree in self.trees
            )
        else:
            predictions_per_tree = [
                self._predict_proba_tree(X, tree) for tree in self.trees
            ]
        distributions = self.calculate_distributions(predictions_per_tree, X.shape[0])
        distributions = np.array(distributions)
        normalize(distributions, copy=False, norm="l1")
        return distributions

    def calculate_prediction_counts(self, predictions):
        """Pick an array of labels predicted by the trees.

        Reorganize it into a dictionary {label: times predicted}
        :param predictions: Array-like which contains the label predicted by each tree
        :return:
        """
        arr = np.array(predictions)
        count_arr = np.bincount(arr)
        prediction_counts = dict()
        for label in np.sort(np.unique(self.y)):
            try:
                prediction_counts[label] = count_arr[label]
            except IndexError:
                prediction_counts[label] = 0
        return prediction_counts

    def calculate_distributions(self, predictions_per_tree, size):
        """Find probability estimates for each class for all cases in X.

        :param predictions_per_tree: Array-like of shape
        [n_instances,[n_tree_estimators,labels]] which contains an array of labels
        predicted by each tree for each instance.
        :param size: Size of X dataset containing the instances to predict
        :return:
        """
        distributions = np.zeros((size, len(np.unique(self.classes_))))
        for index in range(size):
            predicted_classes = list()
            for predict_tree in predictions_per_tree:
                predicted_classes.append(predict_tree[index])
            prediction_counts = self.calculate_prediction_counts(predicted_classes)
            prediction_counts = list(prediction_counts.items())
            prediction_counts_to_array = np.array(prediction_counts)
            sub_distribution = prediction_counts_to_array[
                np.argsort(prediction_counts_to_array[:, 0])
            ][:, 1]
            np.add.at(distributions, index, sub_distribution)
        return distributions
