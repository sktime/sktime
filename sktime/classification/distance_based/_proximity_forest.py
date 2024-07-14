"""Proximity Forest time series classifier.

A decision tree forest which uses distance measures to partition data. B. Lucas and A.
Shifaz, C. Pelletier, L. O'Neill, N. Zaidi, B. Goethals, F. Petitjean and G. Webb
Proximity Forest: an effective and scalable distance-based classifier for time series,
Data Mining and Knowledge Discovery, 33(3): 607-635, 2019
"""

__author__ = ["goastler", "moradabaz"]
__all__ = ["ProximityForest", "ProximityStump", "ProximityTree"]

import math

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.datatypes import convert
from sktime.distances import (
    dtw_distance,
    erp_distance,
    lcss_distance,
    msm_distance,
    wdtw_distance,
)
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.panel.summarize import DerivativeSlopeTransformer

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
    transformer: the transformer to transform uncached data

    Attributes
    ----------
    cache       : location to store transforms seen before for fast look up
    """

    def __init__(self, transformer):
        self.cache = {}
        self.transformer = transformer
        super().__init__()

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

    Parameters
    ----------
    distance_measure: the distance measure to use
    transformer: the transformer to use

    Return
    ------
    a distance measure function with built in transformation
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

    Parameters
    ----------
    distance_measure: callable
        A callable distance measure function to use.
    params: dict
        The parameters to use in the distance measure

    Returns
    -------
    ret: callable
        A distance measure with no parameters
    """

    def distance(instance_a, instance_b):
        return distance_measure(instance_a, instance_b, **params)

    return distance


def numba_wrapper(distance_measure):
    """Wrap a numba distance measure with numpy conversion.

    Converts to 1 column per dimension format. Really would be better if the whole thing
    worked directly with numpy arrays.

    Parameters
    ----------
    distance_measure: callable
        A distance measure to wrap

    Returns
    -------
    ret: callable
        a distance measure which automatically formats data for numba distance
        measures
    """

    def distance(instance_a, instance_b, **params):
        instance_a = convert(instance_a, "nested_univ", "numpyflat")
        instance_b = convert(instance_b, "nested_univ", "numpyflat")
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
        _, class_counts = np.unique(y, return_counts=True)
        # subtract class entropy from current score for each class
        class_counts = np.divide(class_counts, n_instances)
        class_counts = np.power(class_counts, 2)
        sum = np.sum(class_counts)
        return 1 - sum
    else:
        # y is empty, therefore considered pure
        raise ValueError(" y empty")


def dtw_distance_measure_getter(X):
    """Generate the dtw distance measure.

    Parameters
    ----------
    X: dataset to derive parameter ranges from

    Returns
    -------
    ret: distance measure and parameter range dictionary
    """
    return {
        "distance_measure": [numba_wrapper(dtw_distance)],
        "window": stats.uniform(0, 0.25),
    }


def msm_distance_measure_getter(X):
    """Generate the msm distance measure.

    Parameters
    ----------
    X: dataset to derive parameter ranges from

    Returns
    -------
    ret: distance measure and parameter range dictionary
    """
    n_dimensions = 1  # todo use other dimensions
    return {
        "distance_measure": [numba_wrapper(msm_distance)],
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

    Parameters
    ----------
    X: dataset to derive parameter ranges from
    ret: distance measure and parameter range dictionary
    """
    stdp = _stdp(X)
    instance_length = _max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    n_dimensions = 1  # todo use other dimensions
    return {
        "distance_measure": [numba_wrapper(erp_distance)],
        "dim_to_use": stats.randint(low=0, high=n_dimensions),
        "g": stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp),
        "band_size": stats.randint(low=0, high=max_raw_warping_window + 1),
        # scipy stats randint is exclusive on the max value, hence + 1
    }


def lcss_distance_measure_getter(X):
    """Generate the lcss distance measure.

    Parameters
    ----------
    X: dataset to derive parameter ranges from

    Returns
    -------
    ret: distance measure and parameter range dictionary
    """
    stdp = _stdp(X)
    instance_length = _max_instance_length(X)  # todo should this use the max instance
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


# def twe_distance_measure_getter(X):
#     """Generate the twe distance measure.
#
#     :param X: dataset to derive parameter ranges from
#     :returns: distance measure and parameter range dictionary
#     """
#     return {
#         "distance_measure": [cython_wrapper(twe_distance)],
#         "penalty": [
#             0,
#             0.011111111,
#             0.022222222,
#             0.033333333,
#             0.044444444,
#             0.055555556,
#             0.066666667,
#             0.077777778,
#             0.088888889,
#             0.1,
#         ],
#         "stiffness": [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
#     }


def wdtw_distance_measure_getter(X):
    """Generate the wdtw distance measure.

    Parameters
    ----------
    X: dataset to derive parameter ranges from

    Returns
    -------
    ret: distance measure and parameter range dictionary
    """
    return {
        "distance_measure": [numba_wrapper(wdtw_distance)],
        "g": stats.uniform(0, 1),
    }


def euclidean_distance_measure_getter(X):
    """Generate the ed distance measure.

    Parameters
    ----------
    X: dataset to derive parameter ranges from

    Returns
    -------
    ret: distance measure and parameter range dictionary
    """
    return {"distance_measure": [numba_wrapper(dtw_distance)], "w": [0]}


def setup_wddtw_distance_measure_getter(transformer):
    """Generate the wddtw distance measure.

    Bakes the derivative transformer into the dtw distance measure

    Parameters
    ----------
    transformer: the transformer to use

    Returns
    -------
    ret: a getter to produce the distance measure
    """

    def getter(X):
        return {
            "distance_measure": [
                _derivative_distance(numba_wrapper(wdtw_distance), transformer)
            ],
            "g": stats.uniform(0, 1),
        }

    return getter


def setup_ddtw_distance_measure_getter(transformer):
    """Generate the ddtw distance measure.

    Bakes the derivative transformer into the dtw distance measure

    Parameters
    ----------
    transformer: the transformer to use

    Returns
    -------
    ret: a getter to produce the distance measure
    """

    def getter(X):
        return {
            "distance_measure": [
                _derivative_distance(numba_wrapper(dtw_distance), transformer)
            ],
            "w": stats.uniform(0, 0.25),
        }

    return getter


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


# for distance measure getters
TRANSFORMER = _CachedTransformer(DerivativeSlopeTransformer())
DISTANCE_MEASURE_GETTERS = {
    "euclidean": euclidean_distance_measure_getter,
    "dtw": dtw_distance_measure_getter,
    "ddtw": setup_ddtw_distance_measure_getter(TRANSFORMER),
    "wdtw": wdtw_distance_measure_getter,
    "wddtw": setup_wddtw_distance_measure_getter(TRANSFORMER),
    "msm": msm_distance_measure_getter,
    "lcss": lcss_distance_measure_getter,
    "erp": erp_distance_measure_getter,
}


class ProximityStump(BaseClassifier):
    """Proximity Stump class.

    Model a decision stump which uses a distance measure to partition data.

    Parameters
    ----------
    random_state: integer, the random state
    distance_measure: ``None`` (default) or str; if str, one of
        "euclidean", "dtw", "ddtw", "wdtw", "wddtw", "msm", "lcss", "erp"
        distance measure to use
        if ``None``, selects distances randomly from the list of available distances
    verbosity: logging verbosity
    n_jobs: number of jobs to run in parallel *across threads"

    Examples
    --------
    >>> from sktime.classification.distance_based import ProximityStump
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test")  # doctest: +SKIP
    >>> clf = ProximityStump()  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    ProximityStump(...)
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["goastler", "moradabaz"],
        "maintainers": ["goastler", "moradabaz"],
        # estimator type
        # --------------
        "capability:multithreading": True,
        "X_inner_mtype": "nested_univ",  # input in nested dataframe
    }

    def __init__(
        self,
        random_state=None,
        distance_measure=None,
        verbosity=0,
        n_jobs=1,
    ):
        self.random_state = random_state
        self.distance_measure = distance_measure
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
        self.entropy = None
        self._random_object = None
        super().__init__()

    def pick_distance_measure(self):
        """Pick a distance measure.

        Parameters
        ----------
        self : ProximityStump object.

        Returns
        -------
        ret: distance measure
        """
        random_state = check_random_state(self.random_state)

        if self.distance_measure is None:
            distance_measure_getter = random_state.choice(
                list(DISTANCE_MEASURE_GETTERS.values())
            )
        else:
            distance_measure_getter = DISTANCE_MEASURE_GETTERS[self.distance_measure]

        distance_measure_perm = distance_measure_getter(self.X)
        param_perm = pick_rand_param_perm_from_dict(distance_measure_perm, random_state)
        distance_measure = param_perm.pop("distance_measure")
        return distance_predefined_params(distance_measure, **param_perm)

    @staticmethod
    def _distance_to_exemplars_inst(exemplars, instance, distance_measure):
        """Find distance between a given instance and the exemplar instances.

        Parameters
        ----------
        exemplars: the exemplars to use
        instance: the instance to compare to each exemplar
        distance_measure: the distance measure to provide similarity values

        Returns
        -------
        list of distances to each exemplar
        """
        n_exemplars = len(exemplars)
        distances = np.empty(n_exemplars)
        min_distance = math.inf
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

    def get_exemplars(self):
        """Extract exemplars from a dataframe and class value list.

        Parameters
        ----------
        self : ProximityStump, the proximity stump object.

        Returns
        -------
        ret: One exemplar per class
        """
        # find unique class labels
        unique_class_labels = np.unique(self.y)
        n_unique_class_labels = len(unique_class_labels)
        chosen_instances = [None] * n_unique_class_labels
        # for each class randomly choose and instance
        for class_label_index in range(n_unique_class_labels):
            class_label = unique_class_labels[class_label_index]
            # filter class labels for desired class and get indices
            indices = np.argwhere(self.y == class_label)
            # flatten numpy output
            indices = np.ravel(indices)
            # random choice
            index = self._random_object.choice(indices)
            # record exemplar instance and class label
            instance = self.X.iloc[index, :]
            chosen_instances[class_label_index] = instance
        # convert lists to numpy arrays
        return chosen_instances, unique_class_labels

    def _distance_measure(self):
        """Get the distance measure.

        Parameters
        ----------
        self : ProximityStump
            the proximity stump object.

        Returns
        -------
        ret: distance measure
        """
        return self.pick_distance_measure()

    def distance_to_exemplars(self, X):
        """Find distance to exemplars.

        Parameters
        ----------
        X: the dataset containing a list of instances

        Returns
        -------
        ret: 2d numpy array of distances from each instance to each
            exemplar (instance by exemplar)
        """
        if self._threads_to_use > 1:
            parallel = Parallel(self._threads_to_use)
            distances = parallel(
                delayed(self._distance_to_exemplars_inst)(
                    self.X_exemplar, X.iloc[index, :], self._distance_measure()
                )
                for index in range(X.shape[0])
            )
        else:
            distances = [
                self._distance_to_exemplars_inst(
                    self.X_exemplar, X.iloc[index, :], self._distance_measure()
                )
                for index in range(X.shape[0])
            ]
        distances = np.nan_to_num(np.vstack(np.array(distances)))
        return distances

    def _fit(self, X, y):
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
        self.X = _positive_dataframe_indices(X)
        self._random_object = check_random_state(self.random_state)
        self.y = y
        self.X_exemplar, self.y_exemplar = self.get_exemplars()

        return self

    def find_closest_exemplar_indices(self, X):
        """Find the closest exemplar index for each instance in a dataframe.

        Parameters
        ----------
        X: the dataframe containing instances

        Returns
        -------
        ret: 1d numpy array of indices, one for each instance,
            reflecting the index of the closest exemplar
        """
        n_instances = X.shape[0]
        distances = self.distance_to_exemplars(X)
        indices = np.empty(X.shape[0], dtype=int)
        for index in range(n_instances):
            exemplar_distances = distances[index]
            closest_exemplar_index = _arg_min(exemplar_distances, self._random_object)
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
        # if you have custom gain function implemented in the future, you can
        # change the line below
        self.entropy = gini_gain(self.y, self.y_branches)
        return self

    def _predict_proba(self, X) -> np.ndarray:
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
        X = _negative_dataframe_indices(X)
        distances = self.distance_to_exemplars(X)
        ones = np.ones(distances.shape)
        distances = np.add(distances, ones)
        distributions = np.reciprocal(distances)
        normalize(distributions, copy=False, norm="l1")
        return distributions

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {
            "random_state": 0,
        }
        params2 = {"random_state": 42, "distance_measure": "dtw"}
        return [params1, params2]


class ProximityTree(BaseClassifier):
    """Proximity Tree class.

    A decision tree which uses distance measures to partition data.

    Parameters
    ----------
    random_state: int or np.RandomState, default=0
        random seed for the random number generator
    distance_measure: ``None`` (default) or str; if str, one of
        ``euclidean``, ``dtw``, ``ddtw``, ``wdtw``, ``wddtw``, ``msm``,
        ``lcss``, ``erp`` distance measure to use
        if ``None``, selects distances randomly from the list of available distances
    max_depth: int or math.inf, default=math.inf
        maximum depth of the tree
    is_leaf : function, default=pure
        decide when to mark a node as a leaf node
    verbosity: 0 or 1
        number reflecting the verbosity of logging
        0 = no logging, 1 = verbose logging
    n_jobs: int or None, default=1
        number of parallel threads to use while building
    n_stump_evaluations: number of stump evaluations to do if find_stump method is None

    Examples
    --------
    >>> from sktime.classification.distance_based import ProximityTree
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True) # doctest: +SKIP
    >>> clf = ProximityTree(max_depth=2, n_stump_evaluations=1) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    ProximityTree(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["goastler", "moradabaz"],
        "maintainers": ["goastler", "moradabaz"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "X_inner_mtype": "nested_univ",
    }

    def __init__(
        self,
        random_state=None,
        distance_measure=None,
        max_depth=math.inf,
        is_leaf=pure,
        verbosity=0,
        n_jobs=1,
        n_stump_evaluations=5,
    ):
        self.verbosity = verbosity
        self.n_stump_evaluations = n_stump_evaluations
        self.max_depth = max_depth
        self.random_state = random_state
        self.is_leaf = is_leaf
        self.distance_measure = distance_measure
        self.n_jobs = n_jobs
        self.depth = 0

        # below set in fit method
        self.label_encoder = None
        self.stump = None
        self.branches = None
        self.X = None
        self.y = None
        self._random_object = None

        super().__init__()

    def pick_distance_measure(self):
        """Pick a distance measure.

        Parameters
        ----------
        self : ProximityStump object.

        Returns
        -------
        distance measure
        """
        random_state = check_random_state(self.random_state)

        if self.distance_measure is None:
            distance_measure_getter = random_state.choice(
                list(DISTANCE_MEASURE_GETTERS.values())
            )
        else:
            distance_measure_getter = DISTANCE_MEASURE_GETTERS[self.distance_measure]

        distance_measure_perm = distance_measure_getter(self.X)
        param_perm = pick_rand_param_perm_from_dict(distance_measure_perm, random_state)
        distance_measure = param_perm.pop("distance_measure")
        return distance_predefined_params(distance_measure, **param_perm)

    def _fit(self, X, y):
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
        self.X = _positive_dataframe_indices(X)
        self._random_object = check_random_state(self.random_state)
        self.y = y

        self.stump = self.find_stump()
        n_branches = len(self.stump.y_exemplar)
        self.branches = [None] * n_branches
        if self.depth < self.max_depth:
            for index in range(n_branches):
                sub_y = self.stump.y_branches[index]
                if not self.is_leaf(sub_y):
                    sub_tree = ProximityTree(
                        random_state=self.random_state,
                        distance_measure=self.distance_measure,
                        is_leaf=self.is_leaf,
                        verbosity=self.verbosity,
                        max_depth=self.max_depth,
                        n_jobs=self._threads_to_use,
                    )
                    sub_tree.label_encoder = self.label_encoder
                    sub_tree.depth = self.depth + 1
                    self.branches[index] = sub_tree
                    sub_X = self.stump.X_branches[index]
                    sub_tree.fit(sub_X, sub_y)

        return self

    def _predict_proba(self, X) -> np.ndarray:
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
        X = _negative_dataframe_indices(X)
        closest_exemplar_indices = self.stump.find_closest_exemplar_indices(X)
        distribution = np.full((X.shape[0], self.n_classes_), 0.0001)
        for index in range(len(self.branches)):
            indices = np.argwhere(closest_exemplar_indices == index)
            if indices.shape[0] > 0:
                indices = np.ravel(indices)
                sub_tree = self.branches[index]
                if sub_tree is None:
                    sub_distribution = np.full(
                        (1, self.n_classes_), np.finfo(float).eps
                    )
                    class_label = self.stump.y_exemplar[index]
                    sub_distribution[0][self._class_dictionary[class_label]] = 1
                else:
                    sub_X = X.iloc[indices, :]
                    sub_distribution = sub_tree.predict_proba(sub_X)
                if sub_distribution.shape[1] != self.n_classes_:
                    sub_distribution = np.zeros((1, self.n_classes_))
                np.add.at(distribution, indices, sub_distribution)
        normalize(distribution, copy=False, norm="l1")
        return distribution

    def find_stump(self):
        """Find the best stump.

        Returns
        -------
        stump : ProximityStump
            the best stump / split of data of the n attempts.
        """
        if self.n_stump_evaluations < 1:
            raise ValueError("n_stump_evaluations cannot be less than 1")
        stumps = []
        for _ in range(self.n_stump_evaluations):
            stump = ProximityStump(
                random_state=self.random_state,
                distance_measure=self.distance_measure,
                verbosity=self.verbosity,
                n_jobs=self.n_jobs,
            )
            stump.fit(self.X, self.y)
            stump.grow()
            stumps.append(stump)
        return _max(stumps, self._random_object, lambda stump: stump.entropy)

    def get_exemplars(self):
        """Extract exemplars from a dataframe and class value list.

        Parameters
        ----------
        self : ProximityTree
            the proximity tree object.

        Returns
        -------
        ret: One exemplar per class
        """
        # find unique class labels
        unique_class_labels = np.unique(self.y)
        n_unique_class_labels = len(unique_class_labels)
        chosen_instances = [None] * n_unique_class_labels
        # for each class randomly choose and instance
        for class_label_index in range(n_unique_class_labels):
            class_label = unique_class_labels[class_label_index]
            # filter class labels for desired class and get indices
            indices = np.argwhere(self.y == class_label)
            # flatten numpy output
            indices = np.ravel(indices)
            # random choice
            index = self._random_object.choice(indices)
            # record exemplar instance and class label
            instance = self.X.iloc[index, :]
            chosen_instances[class_label_index] = instance
        # convert lists to numpy arrays
        return chosen_instances, unique_class_labels

    def _distance_measure(self):
        """Get the distance measure.

        Parameters
        ----------
        self : ProximityStump
            the proximity tree object.

        Returns
        -------
        ret: distance measure
        """
        return self.pick_distance_measure()

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
        params1 = {"max_depth": 1, "n_stump_evaluations": 1}
        params2 = {"max_depth": 5, "n_stump_evaluations": 2, "distance_measure": "dtw"}
        return [params1, params2]


class ProximityForest(BaseClassifier):
    """Proximity Forest classifier.

    Forest of decision tree which uses distance measures to partition data [1].
    Uses ProximityTree internally.

    Parameters
    ----------
    random_state: int or np.RandomState, default=None
        random seed for the random number generator
    n_estimators: int, default=100
        The number of trees in the forest.
    distance_measure: ``None`` (default) or str; if str, one of
        ``euclidean``, ``dtw``, ``ddtw``, ``wdtw``, ``wddtw``, ``msm``,
        ``lcss``, ``erp`` distance measure to use
        if ``None``, selects distances randomly from the list of available distances
    verbosity: 0 or 1
        number reflecting the verbosity of logging
        0 = no logging, 1 = verbose logging
    max_depth: int or math.inf, default=math.inf
        maximum depth of the tree
    is_leaf: function, default=pure
        function to decide when to mark a node as a leaf node
    n_jobs: int, default=1
        number of jobs to run in parallel *across threads"
    n_stump_evaluations: int, default=5
        number of stump evaluations to do if find_stump method is None

    References
    ----------
    .. [1] Ben Lucas et al., "Proximity Forest: an effective and scalable distance-based
      classifier for time series",vData Mining and Knowledge Discovery, 33(3): 607-635,
      2019 https://arxiv.org/abs/1808.10594
    Java wrapper of authors original
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/distance_based/ProximityForestWrapper.java
    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/distance_based/proximity/ProximityForest.java

    Examples
    --------
    >>> from sktime.classification.distance_based import ProximityForest
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True) # doctest: +SKIP
    >>> clf = ProximityForest(
    ...     n_estimators=2, max_depth=2, n_stump_evaluations=1
    ... ) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    ProximityForest(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["goastler", "moradabaz"],
        "maintainers": ["goastler", "moradabaz"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "X_inner_mtype": "nested_univ",
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "distance",
    }

    def __init__(
        self,
        random_state=None,
        n_estimators=100,
        distance_measure=None,
        verbosity=0,
        max_depth=math.inf,
        is_leaf=pure,
        n_jobs=1,
        n_stump_evaluations=5,
    ):
        self.is_leaf = is_leaf
        self.verbosity = verbosity
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.n_stump_evaluations = n_stump_evaluations
        self.distance_measure = distance_measure

        # set in fit method
        self.label_encoder = None
        self.trees = None
        self.X = None
        self.y = None
        self._random_object = None

        super().__init__()

    def pick_distance_measure(self):
        """Pick a distance measure.

        Parameters
        ----------
        self : ProximityStump object.

        Returns
        -------
        ret: distance measure
        """
        random_state = check_random_state(self.random_state)

        if self.distance_measure is None:
            distance_measure_getter = random_state.choice(
                list(DISTANCE_MEASURE_GETTERS.values())
            )
        else:
            distance_measure_getter = DISTANCE_MEASURE_GETTERS[self.distance_measure]

        distance_measure_perm = distance_measure_getter(self.X)
        param_perm = pick_rand_param_perm_from_dict(distance_measure_perm, random_state)
        distance_measure = param_perm.pop("distance_measure")
        return distance_predefined_params(distance_measure, **param_perm)

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
            print("tree " + str(index) + " building")
        tree = ProximityTree(
            random_state=random_state,
            verbosity=self.verbosity,
            distance_measure=self.distance_measure,
            max_depth=self.max_depth,
            is_leaf=self.is_leaf,
            n_jobs=1,
            n_stump_evaluations=self.n_stump_evaluations,
        )
        tree.fit(X, y)
        return tree

    def _fit(self, X, y):
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
        self.X = _positive_dataframe_indices(X)
        self._random_object = check_random_state(self.random_state)
        self.y = y

        if self._threads_to_use > 1:
            parallel = Parallel(self._threads_to_use)
            self.trees = parallel(
                delayed(self._fit_tree)(
                    X, y, index, self._random_object.randint(0, self.n_estimators)
                )
                for index in range(self.n_estimators)
            )
        else:
            self.trees = [
                self._fit_tree(
                    X, y, index, self._random_object.randint(0, self.n_estimators)
                )
                for index in range(self.n_estimators)
            ]

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

    def _predict_proba(self, X) -> np.ndarray:
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
        X = _negative_dataframe_indices(X)
        if self._threads_to_use > 1:
            parallel = Parallel(self._threads_to_use)
            distributions = parallel(
                delayed(self._predict_proba_tree)(X, tree) for tree in self.trees
            )
        else:
            distributions = [self._predict_proba_tree(X, tree) for tree in self.trees]
        distributions = np.array(distributions)
        distributions = np.sum(distributions, axis=0)
        normalize(distributions, copy=False, norm="l1")
        return distributions

    def get_exemplars(self):
        """Extract exemplars from a dataframe and class value list.

        Parameters
        ----------
        self : ProximityForest
            the proximity forest object.

        Returns
        -------
        ret: One exemplar per class
        """
        # find unique class labels
        unique_class_labels = np.unique(self.y)
        n_unique_class_labels = len(unique_class_labels)
        chosen_instances = [None] * n_unique_class_labels
        # for each class randomly choose and instance
        for class_label_index in range(n_unique_class_labels):
            class_label = unique_class_labels[class_label_index]
            # filter class labels for desired class and get indices
            indices = np.argwhere(self.y == class_label)
            # flatten numpy output
            indices = np.ravel(indices)
            # random choice
            index = self._random_object.choice(indices)
            # record exemplar instance and class label
            instance = self.X.iloc[index, :]
            chosen_instances[class_label_index] = instance
        # convert lists to numpy arrays
        return chosen_instances, unique_class_labels

    def _distance_measure(self):
        """Get the distance measure.

        Parameters
        ----------
        self : ProximityForest
            the proximity forest object.

        Returns
        -------
        ret: distance measure
        """
        return self.pick_distance_measure()

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
            return {"n_estimators": 3, "max_depth": 2, "n_stump_evaluations": 2}
        else:
            param1 = {"n_estimators": 2, "max_depth": 1, "n_stump_evaluations": 1}
            param2 = {
                "n_estimators": 4,
                "max_depth": 2,
                "n_stump_evaluations": 3,
                "distance_measure": "dtw",
            }
            return [param1, param2]


# start of util functions


# find the index of the best value in the array
def arg_bests(array, comparator):
    indices = [0]
    best = array[0]
    for index in range(1, len(array)):
        value = array[index]
        comparison_result = comparator(value, best)
        if comparison_result >= 0:
            if comparison_result > 0:
                indices = []
                best = value
            indices.append(index)
    return indices


# pick values from array at given indices
def _pick_from_indices(array, indices):
    picked = []
    for index in indices:
        picked.append(array[index])
    return picked


# find best values in array
def _bests(array, comparator):
    indices = arg_bests(array, comparator)
    return _pick_from_indices(array, indices)


# find min values in array
def _mins(array, getter=None):
    indices = _arg_mins(array, getter)
    return _pick_from_indices(array, indices)


# find max values in array
def maxs(array, getter=None):
    indices = _arg_maxs(array, getter)
    return _pick_from_indices(array, indices)


# find min value in array, randomly breaking ties
def min(array, rand, getter=None):
    index = _arg_min(array, rand, getter)
    return array[index]


# find index of max value in array, randomly breaking ties
def _arg_max(array, rand, getter=None):
    return rand.choice(_arg_maxs(array, getter))


# find max value in array, randomly breaking ties
def _max(array, rand, getter=None):
    index = _arg_max(array, rand, getter)
    return array[index]


# find best value in array, randomly breaking ties
def _best(array, comparator, rand):
    return rand.choice(_bests(array, comparator))


# find index of best value in array, randomly breaking ties
def _arg_best(array, comparator, rand):
    return rand.choice(arg_bests(array, comparator))


# find index of min value in array, randomly breaking ties
def _arg_min(array, rand, getter=None):
    return rand.choice(_arg_mins(array, getter))


# find indices of best value in array, randomly breaking ties
def _arg_mins(array, getter=None):
    return arg_bests(array, _chain(_less_than, getter))


# find indices of max value in array, randomly breaking ties
def _arg_maxs(array, getter=None):
    return arg_bests(array, _chain(_more_than, getter))


# obtain a value before using in func
def _chain(func, getter=None):
    if getter is None:
        return func
    else:
        return lambda a, b: func(getter(a), getter(b))


# test if a is more than b
def _more_than(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


# test if a is less than b
def _less_than(a, b):
    if a < b:
        return 1
    elif a > b:
        return -1
    else:
        return 0


def _negative_dataframe_indices(X):
    if np.any(X.index >= 0) or len(np.unique(X.index)) > 1:
        X = X.copy(deep=True)
        X.index = np.arange(-1, -len(X.index) - 1, step=-1)
    return X


def _positive_dataframe_indices(X):
    if np.any(X.index < 0) or len(np.unique(X.index)) > 1:
        X = X.copy(deep=True)
        X.index = np.arange(0, len(X.index))
    return X


def _stdp(X):
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
    stdp_val = math.sqrt(sum_sq / num_values - mean**2)
    return stdp_val


def _bin_instances_by_class(X, class_labels):
    bins = {}
    for class_label in np.unique(class_labels):
        bins[class_label] = []
    num_instances = X.shape[0]
    for instance_index in range(0, num_instances):
        instance = X.iloc[instance_index, :]
        class_label = class_labels[instance_index]
        instances_bin = bins[class_label]
        instances_bin.append(instance)
    return bins


def _max_instance_dimension_length(X, dimension):
    num_instances = X.shape[0]
    max = -1
    for instance_index in range(0, num_instances):
        instance = X.iloc[instance_index, dimension]
        if len(instance) > max:
            max = len(instance)
    return max


def _max_instance_length(X):
    # todo use all dimensions / uneven length dataset
    max_length = len(X.iloc[0, 0])
    # max = -1
    # for dimension in range(0, instances.shape[1]):
    #     length = max_instance_dimension_length(instances, dimension)
    #     if length > max:
    #         max = length
    return max_length
