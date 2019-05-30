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

def _derivative_distance(distance_measure, transformer):

    def distance(instance_a, instance_b, **params): # todo limit
        df = pd.DataFrame([instance_a, instance_b])
        df = transformer.transform(X = df)
        instance_a = df.iloc[0, :]
        instance_b = df.iloc[1, :]
        return distance_measure(instance_a, instance_b, **params)

    return distance

def distance_predefined_params(distance_measure, **params):
    def distance(instance_a, instance_b):
        return distance_measure(instance_a, instance_b, **params)

    return distance

def cython_wrapper(distance_measure):
    def distance(instance_a, instance_b, **params):

        # find distance
        instance_a = tabularise(instance_a, return_array = True)  # todo use specific dimension rather than whole
        # thing?
        instance_b = tabularise(instance_b, return_array = True)  # todo use specific dimension rather than whole thing?
        instance_a = np.transpose(instance_a)
        instance_b = np.transpose(instance_b)
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
        return 0.5
    # find gini for parent node
    score = gini(y)
    # sum the children's gini scores
    for index in range(0, len(y_subs)):
        child_class_labels = y_subs[index]
        # ignore empty children
        if len(child_class_labels) > 0:
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
        return 1 - sum
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


def dtw_getter(X):
    instance_length = dataset_properties.max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    max_warping_window_percentage = max_raw_warping_window / instance_length
    return {
        'distance_measure': [cython_wrapper(dtw_distance)],
        'w'               : stats.uniform(0, max_warping_window_percentage)
    }


def setup_ddtw_getter(transformer):
    def ddtw_getter(X):
        instance_length = dataset_properties.max_instance_length(X)  # todo should this use the max instance
        # length for unequal length dataset instances?
        max_raw_warping_window = np.floor((instance_length + 1) / 4)
        max_warping_window_percentage = max_raw_warping_window / instance_length
        return {
            'distance_measure': [_derivative_distance(cython_wrapper(dtw_distance), transformer)],
            'w'               : stats.uniform(0, max_warping_window_percentage)
        }
    return ddtw_getter

def setup_distance_measure_getters(proximity):
    transformer = CachedTransformer(DerivativeSlopeTransformer())
    distance_measure_getters = [
            dtw_getter,
            setup_ddtw_getter(transformer)
            ]
    def pick_rand_distance_measure(proximity):
        random_state = proximity.random_state
        X = proximity.X
        distance_measure_getter = random_state.choice(distance_measure_getters)
        distance_measure_perm = distance_measure_getter(X)
        param_perm = pick_rand_param_perm_from_dict(distance_measure_perm, random_state)
        distance_measure = param_perm['distance_measure']
        del param_perm['distance_measure']
        return distance_predefined_params(distance_measure, **param_perm)
    return pick_rand_distance_measure


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


def negative_dataframe_indices(X):
    if X.index[0] >= 0:
        X = X.copy(deep = True)
        X.index = np.negative(X.index)
        X.index -= 1
    return X

def positive_dataframe_indices(X):
    if X.index[0] < 0:
        X = X.copy(deep = True)
        X.index = np.abs(X.index)
    return X

class PS(BaseClassifier):
    def __init__(self,
                 random_state = None,
                 get_exemplars = get_one_exemplar_per_class,
                 setup_distance_measure = setup_distance_measure_getters,
                 get_distance_measure = None,
                 distance_measure = None,
                 get_entropy = gini_gain,
                 verbosity = 0,
                 ):
        self.setup_distance_measure = setup_distance_measure
        self.random_state = random_state
        self.get_distance_measure = get_distance_measure
        self.distance_measure = distance_measure
        self.pick_exemplars = get_exemplars
        self.get_entropy = get_entropy
        self.verbosity = verbosity
        # set in fit
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
                    distance = self.distance_measure(instance, exemplar) #, min_distance)
                if distance < min_distance:
                    min_distance = distance
                distances[instance_index, exemplar_index] = distance
        return distances

    def fit(self, X, y):
        # todo checks
        self.X = positive_dataframe_indices(X)
        self.y = y
        self.random_state = check_random_state(self.random_state)
        # setup label encoding if not already
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        if self.distance_measure is None:
            if self.get_distance_measure is None:
                self.get_distance_measure = self.setup_distance_measure(self)
            self.distance_measure = self.get_distance_measure(self)
        self.X_exemplar, self.y_exemplar = self.pick_exemplars(self)
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
        self.X = negative_dataframe_indices(X)
        distances = self.distance_to_exemplars(X)
        ones = np.ones(distances.shape)
        distances = np.add(distances, ones)
        distributions = np.divide(ones, distances)
        normalize(distributions, copy = False, norm = 'l1')
        return distributions


class PT(BaseClassifier):

    def __init__(self,
                 # note: any changes of these params must be reflected in the fit method for building trees
                 random_state = None,
                 get_exemplars = get_one_exemplar_per_class,
                 distance_measure = None,
                 get_distance_measure = None,
                 setup_distance_measure = setup_distance_measure_getters,
                 get_entropy = gini_gain,
                 num_stump_evaluations = 5,
                 is_leaf = pure,
                 verbosity = 0,
                 label_encoder = None
                 ):
        self.verbosity = verbosity
        self.get_distance_measure = distance_measure
        self.random_state = random_state
        self.is_leaf = is_leaf
        self.get_distance_measure = get_distance_measure
        self.setup_distance_measure = setup_distance_measure
        self.get_exemplars = get_exemplars
        self.label_encoder = label_encoder # todo labelenc in ps
        self.get_entropy = get_entropy
        self.num_stump_evaluations = num_stump_evaluations
        # below set in fit method
        self.distance_measure = None
        self.stump = None
        self.branches = None
        self.classes_ = None

    def fit(self, X, y):
        # print('building tree on ' + str(y))
        # todo checks
        self.X = positive_dataframe_indices(X)
        self.y = y
        self.random_state = check_random_state(self.random_state)
        # setup label encoding if not already
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        if self.distance_measure is None:
            if self.get_distance_measure is None:
                self.get_distance_measure = self.setup_distance_measure(self)
            self.distance_measure = self.get_distance_measure(self)
        stumps = []
        for index in range(0, self.num_stump_evaluations):
            stump = PS(
                    random_state = self.random_state,
                    get_exemplars = self.get_exemplars,
                    distance_measure = self.distance_measure,
                    setup_distance_measure = self.setup_distance_measure,
                    get_distance_measure = self.get_distance_measure,
                    get_entropy = self.get_entropy,
                    verbosity = self.verbosity,
                    )
            stump.fit(X, y)
            stump.grow()
            stumps.append(stump)
        self.stump = comparison.max(stumps, self.random_state, lambda stump: stump.entropy)
        self.branches = []
        # print('branches: ' + str(self.stump.y_branches))
        for index in range(0, len(self.stump.y_exemplar)):
            y = self.stump.y_branches[index]
            sub_tree = None
            if not self.is_leaf(y):
                X = self.stump.X_branches[index]
                sub_tree = PT(
                 random_state = self.random_state,
                 get_exemplars = self.get_exemplars,
                 distance_measure = self.distance_measure,
                 setup_distance_measure = self.setup_distance_measure,
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
        self.X = negative_dataframe_indices(X)
        closest_exemplar_indices = self.stump.find_closest_exemplar_indices(X)
        num_classes = len(self.label_encoder.classes_)
        predict_probas = np.zeros((X.shape[0], num_classes))
        for index in range(0, len(self.branches)):
            indices = np.argwhere(closest_exemplar_indices == index)
            if indices.shape[0] > 0:
                indices = np.ravel(indices)
                sub_tree = self.branches[index]
                if sub_tree is None:
                    sub_predict_probas = np.zeros(num_classes)
                    class_label = self.stump.y_exemplar[index]
                    sub_predict_probas[class_label] = 1
                else:
                    sub_X = X.iloc[indices, :]
                    sub_predict_probas = sub_tree.predict_proba(sub_X)
                np.add.at(predict_probas, indices, sub_predict_probas)
        normalize(predict_probas, copy = False, norm = 'l1')
        return predict_probas

    # todo get params avoid func pointer - use name
    # todo set params use func name or func pointer
    # todo classes_ var in classifiers
    # todo constructor accept str name func / pointer

class PF(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 num_trees = 100,
                 label_encoder = None,
                 distance_measure = None,
                 get_distance_measure = None,
                 verbosity = 0,
                 setup_distance_measure = setup_distance_measure_getters,
                 ):
        self.verbosity = verbosity
        self.random_state = random_state
        self.num_trees = num_trees
        self.label_encoder = label_encoder
        self.get_distance_measure = get_distance_measure
        self.setup_distance_measure = setup_distance_measure
        self.distance_measure = distance_measure
        # set in fit method
        self.trees = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        # todo checks
        self.X = positive_dataframe_indices(X)
        self.y = y
        self.random_state = check_random_state(self.random_state)
        # setup label encoding if not already
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        if self.distance_measure is None:
            if self.get_distance_measure is None:
                self.get_distance_measure = self.setup_distance_measure(self)
            self.distance_measure = self.get_distance_measure(self)
        self.trees = []
        for index in range(0, self.num_trees):
            if self.verbosity > 0:
                print('building tree ' + str(index))
            tree = PT(
                    random_state = self.random_state,
                    verbosity =  self.verbosity,
                    label_encoder = self.label_encoder,
                    distance_measure = self.distance_measure,
                    setup_distance_measure = self.setup_distance_measure,
                    get_distance_measure = self.get_distance_measure,
                      )
            self.trees.append(tree)
            tree.fit(X, y)

    def predict_proba(self, X):
        self.X = negative_dataframe_indices(X)
        predict_probas = np.zeros((X.shape[0], len(self.label_encoder.classes_)))
        for tree in self.trees:
            tree_predict_probas = tree.predict_probas(X)
            predict_probas = np.add(predict_probas, tree_predict_probas)
        normalize(predict_probas, copy = False, norm = 'l1')
        return predict_probas
