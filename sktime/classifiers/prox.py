import warnings

import sklearn
import sktime.utils.transformations

from .base import BaseClassifier
import numpy as np
from sklearn.utils import check_random_state
from ..utils import dataset_properties
from ..transformers.series_to_series import DerivativeSlopeTransformer, CachedTransformer
from ..distances import (
    dtw_distance, erp_distance, lcss_distance, msm_distance, wdtw_distance, twe_distance,
    )
from scipy import stats
from ..utils import comparison
from sklearn.preprocessing import normalize, LabelEncoder
from ..utils.transformations import tabularise
import pandas as pd
from sklearn.base import clone

_derivative_transformer = CachedTransformer(DerivativeSlopeTransformer())

def _derivative_distance(distance_measure, transformer):

    def distance(instance_a, instance_b): # todo limit
        df = pd.DataFrame([instance_a, instance_b])
        df = transformer.transform(df)
        instance_a = df.iloc[0, :]
        instance_b = df.iloc[1, :]
        return distance_measure(instance_a, instance_b)

    return distance

def distance_predefined_params(distance_measure, **params):
    def distance(instance_a, instance_b):
        return distance_measure(instance_a, instance_b, **params)

    return distance

def pure(y):
    '''
    test whether a set of class labels are pure (i.e. all the same)
    ----
    Parameters
    ----
    y : 1d numpy array
        array of class labels
    ----
    Returns
    ----
    result : boolean
        whether the set of class labels is pure
    '''
    # get unique class labels
    unique_class_labels = np.unique(y)
    # if more than 1 unique then not pure
    return len(unique_class_labels) <= 1


def gini_gain(y, y_subs):
    '''
    get gini score of a split, i.e. the gain from parent to children
    ----
    Parameters
    ----
    y : 1d numpy array
        array of class labels at parent
    y_subs : list of 1d numpy array
        list of array of class labels, one array per child
    ----
    Returns
    ----
    score : float
        gini score of the split from parent class labels to children. Note the gini score is scaled to be between 0
        and 1. 1 == pure, 0 == not pure
    '''
    # find number of instances overall
    parent_num_instances = y.shape[0]
    # if parent has no instances then is pure
    if parent_num_instances == 0:
        for child in y_subs:
            if len(child) > 0:
                raise ValueError('children populated but parent empty')
        return 1
    # find gini for parent node
    score = gini(y)
    # if parent is pure then children will also be pure
    if score == 1:
        warnings.warn('parent gini 0')
        return 1
    # sum the children's gini scores
    for index in range(0, len(y_subs)):
        child_class_labels = y_subs[index]
        # ignore empty children
        if len(y_subs) > 0:
            # find gini score for this child
            child_score = gini(child_class_labels)
            # weight score by proportion of instances at child compared to parent
            child_size = len(child_class_labels)
            child_score *= (child_size / parent_num_instances)
            # add to cumulative sum
            score -= child_score
    return score


def gini(y):
    '''
    get gini score at a specific node
    ----
    Parameters
    ----
    y : 1d numpy array
        array of class labels
    ----
    Returns
    ----
    score : float
        gini score for the set of class labels (i.e. how pure they are). 1 == pure, 0 == not pure
    '''
    # get number instances at node
    num_instances = y.shape[0]
    if num_instances > 0:
        # count each class
        unique_class_labels, class_counts = np.unique(y, return_counts = True)
        # subtract class entropy from current score for each class
        class_counts = np.divide(class_counts, num_instances)
        class_counts = np.power(class_counts, 2)
        sum = np.sum(class_counts)
        sum = 1 - sum
        return sum
    else:
        # y is empty, therefore considered pure
        raise ValueError(' y empty')


def get_one_exemplar_per_class(proximity_tree):
    X = proximity_tree.X
    y = proximity_tree.y
    random_state = proximity_tree.random_state
    # find unique class labels
    unique_class_labels = np.unique(y)
    num_unique_class_labels = len(unique_class_labels)
    chosen_instances = [None] * num_unique_class_labels
    # for each class randomly choose and instance
    for class_label_index in range(0, num_unique_class_labels):
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


def get_all_distance_measures_param_pool(X):
    '''
    find parameter pool for all available distance measures
    ----
    Parameters
    ----
    X : panda dataframe
        instances representing a dataset
    dimension : int
        index of dimension to use
    ----
    Returns
    ----
    param_pool : list of dicts
        list of dictionaries to pick distance measures and corresponding parameters from. This should be in the same
        format as sklearn's GridSearchCV parameters
    '''
    # find dataset properties
    # todo any better way to not recalculate stdp and inst length every time?
    num_dimensions = 1  # todo use other dimensions
    instance_length = dataset_properties.max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    max_warping_window_percentage = max_raw_warping_window / instance_length
    stdp = dataset_properties.stdp(X)
    # setup param pool dictionary array (same structure as sklearn's GridSearchCV params!)
    # get keys for dict
    param_pool = [
            {
                    'distance_measure': [dtw_distance],
                    'w'   : stats.uniform(0, max_warping_window_percentage)
                    },
            # {
            #         dm_key: [dtw_distance],
            #         tf_key: [
            #                 _derivative_transformer
            #                 ],
            #         'w'   : stats.uniform(0, max_warping_window_percentage)
            #         },
            # {
            #         dm_key: [wdtw_distance],
            #         'g'   : stats.uniform(0,
            #                               1)
            #         },
            # {
            #         dm_key: [wdtw_distance],
            #         tf_key: [
            #                 _derivative_transformer
            #                 ],
            #         'g'   : stats.uniform(0,
            #                               1)
            #         },
            # {
            #         dm_key      : [lcss_distance],
            #         'dim_to_use': stats.randint(low = 0, high = num_dimensions),
            #         'epsilon'   : stats.uniform(0.2 * stdp, stdp - 0.2 * stdp),
            #         'delta'     : stats.randint(low = 0, high = max_raw_warping_window +
            #                                                     1)  # scipy stats randint
            #         # is exclusive on the max value, hence + 1
            #         },
            # {
            #         dm_key      : [erp_distance],
            #         'dim_to_use': stats.randint(low = 0, high = num_dimensions),
            #         'g'         : stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp),
            #         'band_size' : stats.randint(low = 0, high = max_raw_warping_window + 1)
            #         # scipy stats randint is exclusive on the max value, hence + 1
            #         },
            # {
            #         dm_key     : [twe_distance],
            #         'penalty'  : [0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667,
            #                       0.077777778, 0.088888889, 0.1],
            #         'stiffness': [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            #         },
            # {
            #         dm_key      : [msm_distance],
            #         'dim_to_use': stats.randint(low = 0, high = num_dimensions),
            #         'c'         : [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325,
            #                        0.03625, 0.04, 0.04375, 0.0475, 0.05125,
            #                        0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775,
            #                        0.08125, 0.085, 0.08875, 0.0925, 0.09625,
            #                        0.1, 0.136, 0.172, 0.208,
            #                        0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496,
            #                        0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748,
            #                        0.784, 0.82, 0.856,
            #                        0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8,
            #                        3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68,
            #                        6.04, 6.4, 6.76, 7.12,
            #                        7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2,
            #                        20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46,
            #                        49.6, 53.2, 56.8, 60.4,
            #                        64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4,
            #                        100]
            #         },
            ]
    return param_pool


def pick_rand_param_perm_from_dict(param_pool, random_state):
    '''
    pick a parameter permutation given a list of dictionaries contain potential values OR a list of values OR a
    distribution of values (a distribution must have the .rvs() function to sample values)
    ----------
    param_pool : list of dicts OR list OR distribution
        parameters in the same format as GridSearchCV from scikit-learn. example:
        param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}],
          'kernel': ['rbf']},
         ]
    Returns
    -------
    param_perm : dict
        distance measure and corresponding parameters in dictionary format
    '''
    # construct empty permutation
    param_perm = {}
    # for each parameter
    for param_name, param_values in param_pool.items():
        # if it is a list
        if isinstance(param_values, list):
            # randomly pick a value
            param_value = param_values[random_state.randint(len(param_values))]
            # if the value is another dict then get a random parameter permutation from that dict (recursive over
            # 2 funcs)
            # if isinstance(param_value, dict): # no longer require recursive param perms
            #     param_value = _pick_param_permutation(param_value, random_state)
        # else if parameter is a distribution
        elif hasattr(param_values, 'rvs'):
            # sample from the distribution
            param_value = param_values.rvs(random_state = random_state)
        else:
            # otherwise we don't know how to obtain a value from the parameter
            raise Exception('unknown type of parameter pool')
        # add parameter name and value to permutation
        param_perm[param_name] = param_value
    return param_perm


def pick_rand_param_perm_from_list(params, random_state):
    '''
    get a random parameter permutation providing a distance measure and corresponding parameters
    ----------
    params : list of dicts
        parameters in the same format as GridSearchCV from scikit-learn. example:
        param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}], 'kernel': ['rbf']},
         ]
    Returns
    -------
    permutation : dict
        distance measure and corresponding parameters in dictionary format
    '''
    #
    param_pool = random_state.choice(params)
    permutation = pick_rand_param_perm_from_dict(param_pool, random_state)
    return permutation


def get_rand_distance_measure(proximity_tree):
    random_state = proximity_tree.random_state
    pool = get_all_distance_measures_param_pool(proximity_tree.X)
    permutation = pick_rand_param_perm_from_list(pool, random_state)
    distance_measure = permutation['distance_measure']
    params = permutation.copy()
    del params['distance_measure']
    return distance_predefined_params(distance_measure, **params)


def best_of_n_splits(n):
    if n < 1:
        raise ValueError('n cannot be less than 1')

    def split(proximity_tree):
        trees = []
        for i in range(0, n):
            tree = proximity_tree.clone(deep = False)
            tree.split()
            trees.append(tree)
        best_tree = comparison.best(trees, lambda a, b: a.gain - b.gain, proximity_tree.random_state)
        return best_tree

    return split


class PS(BaseClassifier):
    def __init__(self,
                 random_state = None,
                 get_exemplars = get_one_exemplar_per_class,
                 get_distance_measure = get_rand_distance_measure,
                 get_entropy = gini_gain,
                 ):
        self.random_state = random_state
        self.pick_exemplars = get_exemplars
        self.get_entropy = get_entropy
        self.get_distance_measure = get_distance_measure
        # set in fit
        self.distance_measure = None
        self.y_exemplar = None
        self.X_exemplar = None
        self.X_branches = None
        self.y_branches = None
        self.X = None
        self.y = None
        self.entropy = None

    def distance_to_exemplars(self, X):
        num_instances = X.shape[0]
        num_exemplars = len(self.y_exemplar)
        distances = np.empty((num_instances, num_exemplars))
        for instance_index in range(0, num_instances):
            instance = X.iloc[instance_index, :]
            min_distance = np.math.inf
            for exemplar_index in range(0, num_exemplars):
                exemplar = self.X_exemplar[exemplar_index]
                if exemplar.name == instance.name:
                    distance = 0
                else:
                    distance = self._find_distance(instance, exemplar, min_distance)
                if distance < min_distance:
                    min_distance = distance
                distances[instance_index, exemplar_index] = distance
        return distances

    def fit(self, X, y):
        # todo checks
        self.X = X
        self.y = y
        self.X_exemplar, self.y_exemplar = self.pick_exemplars(self)
        self.distance_measure = self.get_distance_measure(self)
        return self

    def find_closest_exemplar_indices(self, X):
        num_instances = X.shape[0]
        distances = self.distance_to_exemplars(X)
        indices = np.empty(X.shape[0], dtype = int)
        for index in range(0, num_instances):
            exemplar_distances = distances[index]
            closest_exemplar_index = comparison.arg_min(exemplar_distances, self.random_state)
            indices[index] = closest_exemplar_index
        return indices

    def grow(self):
        num_exemplars = len(self.y_exemplar)
        indices = self.find_closest_exemplar_indices(self.X)
        self.X_branches = [None] * num_exemplars
        self.y_branches = [None] * num_exemplars
        for index in range(0, num_exemplars):
            instance_indices = np.argwhere(indices == index)
            instance_indices = np.ravel(instance_indices)
            self.X_branches[index] = self.X.iloc[instance_indices, :]
            y = np.take(self.y, instance_indices)
            self.y_branches[index] = y
        self.entropy = self.get_entropy(self.y, self.y_branches)
        return self

    def predict_proba(self, X):
        distances = self.distance_to_exemplars(X)
        ones = np.ones(distances.shape)
        distributions = np.divide(ones, distances)
        normalize(distributions, copy = False, norm = 'l1')
        return distributions

    def _find_distance(self, instance_a, instance_b, limit):
        # find distance
        instance_a = tabularise(instance_a, return_array = True)  # todo use specific dimension rather than whole
        # thing?
        instance_b = tabularise(instance_b, return_array = True)  # todo use specific dimension rather than whole thing?
        instance_a = np.transpose(instance_a)
        instance_b = np.transpose(instance_b)
        return self.distance_measure(instance_a, instance_b) # todo limit


class PT(BaseClassifier):

    def __init__(self,
                 # note: any changes of these params must be reflected in the fit method for building trees
                 random_state = None,
                 get_exemplars = get_one_exemplar_per_class,
                 get_distance_measure = get_rand_distance_measure,
                 get_entropy = gini_gain,
                 num_stump_evaluations = 5,
                 is_leaf = pure,
                 verbosity = 0,
                 label_encoder = None
                 ):
        self.verbosity = verbosity
        self.random_state = random_state
        self.is_leaf = is_leaf
        self.get_exemplars = get_exemplars
        self.label_encoder = label_encoder # todo labelenc in ps
        self.get_entropy = get_entropy
        self.get_distance_measure = get_distance_measure
        self.num_stump_evaluations = num_stump_evaluations
        # below set in fit method
        self.stump = None
        self.branches = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        stumps = []
        for index in range(0, self.num_stump_evaluations):
            stump = PS(
                    random_state = self.random_state,
                    get_exemplars = self.get_exemplars,
                    get_distance_measure = self.get_distance_measure,
                    get_entropy = self.get_entropy,
                    )
            stump.fit(X, y)
            stump.grow()
            stumps.append(stump)
        self.stump = comparison.max(stumps, self.random_state, lambda stump: stump.entropy)
        self.branches = []
        for index in range(0, len(self.stump.y_exemplar)):
            y = self.stump.y_branches[index]
            sub_tree = None
            if not self.is_leaf(y):
                X = self.stump.X_branches[index]
                sub_tree = PT(
                 random_state = self.random_state,
                 get_exemplars = self.get_exemplars,
                 get_distance_measure = self.get_distance_measure,
                 get_entropy = self.get_entropy,
                 num_stump_evaluations = self.num_stump_evaluations,
                 is_leaf = self.is_leaf,
                 verbosity = self.verbosity,
                 label_encoder = self.label_encoder)
            self.branches.append(sub_tree)
            if sub_tree is not None:
                sub_tree.fit(X, y)
        return self

    def predict_proba(self, X):
        closest_exemplar_indices = self.stump.find_closest_exemplar_indices(X) # todo use numpy argwhere
        num_classes = len(self.label_encoder.classes_)
        predict_probas = np.zeros((X.shape[0], num_classes))
        for index in range(0, len(self.branches)):
            indices = np.argwhere(closest_exemplar_indices == index)
            if indices.shape[0] > 0:
                indices = np.ravel(indices)
                sub_tree = self.branches[index]
                if sub_tree is None:
                    sub_predict_probas = np.zeros(num_classes)
                    sub_predict_probas[index] = 1
                else:
                    sub_X = X.iloc[indices, :]
                    sub_predict_probas = sub_tree.predict_proba(sub_X)
                np.add.at(predict_probas, indices, sub_predict_probas)
        return predict_probas

    # todo get params avoid func pointer - use name
    # todo set params use func name or func pointer
    # todo classes_ var in classifiers
    # todo constructor accept str name func / pointer
