# Proximity Forest: An effective and scalable distance-based classifier for time series
#
# author: George Oastler (linkedin.com/goastler)
#
# paper link: https://arxiv.org/abs/1808.10594
# bibtex reference:
# @article{DBLP:journals/corr/abs-1808-10594,
#   author    = {Benjamin Lucas and
#                Ahmed Shifaz and
#                Charlotte Pelletier and
#                Lachlan O'Neill and
#                Nayyar A. Zaidi and
#                Bart Goethals and
#                Fran{\c{c}}ois Petitjean and
#                Geoffrey I. Webb},
#   title     = {Proximity Forest: An effective and scalable distance-based classifier
#                for time series},
#   journal   = {CoRR},
#   volume    = {abs/1808.10594},
#   year      = {2018},
#   url       = {http://arxiv.org/abs/1808.10594},
#   archivePrefix = {arXiv},
#   eprint    = {1808.10594},
#   timestamp = {Mon, 03 Sep 2018 13:36:40 +0200},
#   biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1808-10594},
#   bibsource = {dblp computer science bibliography, https://dblp.org}
# }
#
# todo unit tests
# todo logging package rather than print to screen
# todo parallelise (specifically tree building, each branch is an independent unit of work)
# todo transformer dist meas str
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import check_random_state, Parallel

from .base import BaseClassifier
from ..distances import (dtw_distance, erp_distance, lcss_distance, msm_distance, twe_distance, wdtw_distance)
from ..transformers.series_to_series import CachedTransformer, DerivativeSlopeTransformer
from ..utils import comparison, dataset_properties
from ..utils.transformations import tabularise


def _derivative_distance(distance_measure, transformer):
    def distance(instance_a, instance_b, **params):  # todo limit
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


def dtw_distance_measure_getter(X):
    return {
            'distance_measure': [cython_wrapper(dtw_distance)],
            'w'               : stats.uniform(0, 0.25)
            }


def msm_distance_measure_getter(X):
    num_dimensions = 1  # todo use other dimensions
    return {
            'distance_measure': [cython_wrapper(msm_distance)],
            'dim_to_use'      : stats.randint(low = 0, high = num_dimensions),
            'c'               : [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325,
                                 0.03625, 0.04, 0.04375, 0.0475, 0.05125,
                                 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775,
                                 0.08125, 0.085, 0.08875, 0.0925, 0.09625,
                                 0.1, 0.136, 0.172, 0.208,
                                 0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496,
                                 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748,
                                 0.784, 0.82, 0.856,
                                 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8,
                                 3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68,
                                 6.04, 6.4, 6.76, 7.12,
                                 7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2,
                                 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46,
                                 49.6, 53.2, 56.8, 60.4,
                                 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4,
                                 100]
            }


def erp_distance_measure_getter(X):
    stdp = dataset_properties.stdp(X)
    instance_length = dataset_properties.max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    num_dimensions = 1  # todo use other dimensions
    return {
            'distance_measure': [cython_wrapper(erp_distance)],
            'dim_to_use'      : stats.randint(low = 0, high = num_dimensions),
            'g'               : stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp),
            'band_size'       : stats.randint(low = 0, high = max_raw_warping_window + 1)
            # scipy stats randint is exclusive on the max value, hence + 1
            }


def lcss_distance_measure_getter(X):
    stdp = dataset_properties.stdp(X)
    instance_length = dataset_properties.max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    num_dimensions = 1  # todo use other dimensions
    return {
            'distance_measure': [cython_wrapper(lcss_distance)],
            'dim_to_use'      : stats.randint(low = 0, high = num_dimensions),
            'epsilon'         : stats.uniform(0.2 * stdp, stdp - 0.2 * stdp),
            'delta'           : stats.randint(low = 0, high = max_raw_warping_window +
                                                              1)  # scipy stats randint
            # is exclusive on the max value, hence + 1
            }


def twe_distance_measure_getter(X):
    return {
            'distance_measure': [cython_wrapper(twe_distance)],
            'penalty'         : [0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667,
                                 0.077777778, 0.088888889, 0.1],
            'stiffness'       : [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            }


def wdtw_distance_measure_getter(X):
    return {
            'distance_measure': [cython_wrapper(wdtw_distance)],
            'g'               : stats.uniform(0,
                                              1)
            }


def euclidean_distance_measure_getter(X):
    return {
            'distance_measure': [cython_wrapper(dtw_distance)],
            'w'               : [0]
            }


def setup_wddtw_distance_measure_getter(transformer):
    def getter(X):
        return {
                'distance_measure': [_derivative_distance(cython_wrapper(wdtw_distance), transformer)],
                'g'               : stats.uniform(0,
                                                  1)
                }

    return getter


def setup_ddtw_distance_measure_getter(transformer):
    def getter(X):
        return {
                'distance_measure': [_derivative_distance(cython_wrapper(dtw_distance), transformer)],
                'w'               : stats.uniform(0,
                                                  0.25)
                }

    return getter


def setup_distance_measure_getters(proximity):
    transformer = CachedTransformer(DerivativeSlopeTransformer())
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


_parallel = Parallel(n_jobs = -1)

class PS(BaseClassifier):
    '''
        proximity tree classifier of depth 1 - in other words, a k=1 nearest neighbour classifier with neighbourhood
        limited
        to x exemplar instances
        ----
        Parameters
        ----
        pick_exemplars_method : callable
            Method to pick exemplars from a set of instances and class labels
        param_perm : dict
            a dictionary containing a distance measure and corresponding parameter
        gain_method : callable
            a method to calculate the gain of this split / stump
        label_encoder : LabelEncoder
            a label encoder, can be pre-populated
        random_state : numpy RandomState
            a random state for sampling random numbers
        verbosity : int
            level of verbosity in output
        dimension : int
            dimension of the dataset to use. Defaults to zero for univariate datasets.
        ----
        Attributes
        ----
        exemplar_instances : panda dataframe
            the chosen exemplar instances
        exemplar_class_labels : numpy 1d array
            array of class labels corresponding to the exemplar instances
            the exemplar instances class labels
        remaining_instances : panda dataframe
            the remaining instances after exemplars have been removed
        remaining_class_labels : numpy 1d array
            array of class labels corresponding to the exemplar instances
            the remaining instances class labels after picking exemplars
        branch_instances : list of panda dataframes
            list of dataframes of instances, one for each child of this stump. I.e. if a stump splits into two children,
            there will be a list of dataframes of length two. branch_instance[0] will contain all train instances
            closest to exemplar 0, branch_instances[1] contains all train instances closest to exemplar 1,
            etc. Exemplars are in the exemplar_instances variable
        branch_class_labels: list of numpy 1d arrays
            similar to branch_instances, but contains the class labels of the instances closest to each exemplar
        distance_measure_param_perm: dict
            parameters to pass to the distance measure method
        distance_measure: callable
            the distance measure to use for measure similarity between instances
        gain: float
            the gain of this stump
        label_encoder : LabelEncoder
            a label encoder, can be pre-populated
        classes_ :
            pointer to the label_encoder classes_
        '''

    __author__ = 'George Oastler (linkedin.com/goastler)'

    def __init__(self,
                 random_state = None,
                 get_exemplars = get_one_exemplar_per_class,
                 setup_distance_measure = setup_distance_measure_getters,
                 get_distance_measure = None,
                 distance_measure = None,
                 get_entropy = gini_gain,
                 verbosity = 0,
                 label_encoder = None,
                 ):
        self.setup_distance_measure = setup_distance_measure
        self.random_state = random_state
        self.get_distance_measure = get_distance_measure
        self.distance_measure = distance_measure
        self.pick_exemplars = get_exemplars
        self.get_entropy = get_entropy
        self.verbosity = verbosity
        self.label_encoder = label_encoder
        # set in fit
        self.y_exemplar = None
        self.X_exemplar = None
        self.X_branches = None
        self.y_branches = None
        self.X = None
        self.y = None
        self.classes_ = None
        self.entropy = None

    def _distance_to_exemplars_inst(self, X, start, end):
        X = X.iloc[range(start, end + 1)]
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
                    distance = self.distance_measure(instance, exemplar)  # , min_distance)
                if distance < min_distance:
                    min_distance = distance
                distances[instance_index, exemplar_index] = distance
        return distances

    # def _distance_to_exemplars_inst(self, X, instance_index):
    #     num_exemplars = len(self.y_exemplar)
    #     instance = X.iloc[instance_index, :]
    #     min_distance = np.math.inf
    #     distances = np.empty(num_exemplars)
    #     for exemplar_index in range(0, num_exemplars):
    #         exemplar = self.X_exemplar[exemplar_index]
    #         if exemplar.name == instance.name:
    #             distance = 0
    #         else:
    #             distance = self.distance_measure(instance, exemplar)  # , min_distance)
    #         if distance < min_distance:
    #             min_distance = distance
    #         distances[exemplar_index] = distance
    #     return distances

    def _batch(self, X):
        num_instances = X.shape[0]
        n_jobs = _parallel.n_jobs
        if n_jobs < 0:
            n_jobs = _parallel._effective_n_jobs()
        start = 0
        batch_size = num_instances / n_jobs
        end = start + batch_size - 1
        start = int(start)
        end = int(end)
        for i in range(0, n_jobs - 1):
            yield start, end
            start += batch_size
            end += batch_size
            start = int(start)
            end = int(end)
        end = num_instances - 1
        start = int(start)
        end = int(end)
        yield start, end

    def distance_to_exemplars(self, X):
        # todo checks

        distances = _parallel(delayed(self._distance_to_exemplars_inst)(X, start, end) for start, end in self._batch(X))
        distances = np.vstack(distances)
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
        # todo checks
        self.X = negative_dataframe_indices(X)
        distances = self.distance_to_exemplars(X)
        ones = np.ones(distances.shape)
        distances = np.add(distances, ones)
        distributions = np.divide(ones, distances)
        normalize(distributions, copy = False, norm = 'l1')
        return distributions


class PT(BaseClassifier):
    '''
        proximity tree classifier using proximity stumps at each tree node to split data into branches.
        ----
        Parameters
        ----
        pick_exemplars_method : callable
            Method to pick exemplars from a set of instances and class labels
        param_pool : list of dicts
            a list of dictionaries containing a distance measure and corresponding parameter sources (distribution or
            predefined value)
        gain_method : callable
            a method to calculate the gain of this split / stump
        label_encoder : LabelEncoder
            a label encoder, can be pre-populated
        random_state : numpy RandomState
            a random state for sampling random numbers
        dimension : int
            dimension of the dataset to use. Defaults to zero for univariate datasets.
        verbosity : int
            level of verbosity in output
        num_stump_evaluations : int
            the number of proximity stumps to produce at each node. Each stump has a random distance measure and
            distance
            measure parameter set. The stump with the best gain is used to split the data.
        is_leaf_method : callable
            a method which takes a split of data and produces a boolean value indicating whether the tree should
            continue
            splitting.
        ----
        Attributes
        ----
        level : int
            the level of the current tree. Each tree is made up of a collection of trees, one for each branch. Each one
            of these trees are level + 1 deep. The level begins on 0.
        branches : array of trees
            trees corresponding to each branch output of the proximity stump.
        stump : ProximityStump
            the proximity stump used to split the data at this node.
        label_encoder : LabelEncoder
            a label encoder, can be pre-populated
        classes_ :
            pointer to the label_encoder classes_
        '''

    __author__ = 'George Oastler (linkedin.com/goastler)'

    def __init__(self,
                 # note: any changes of these params must be reflected in the fit method for building trees / clones
                 random_state = None,
                 get_exemplars = get_one_exemplar_per_class,
                 distance_measure = None,
                 get_distance_measure = None,
                 setup_distance_measure = setup_distance_measure_getters,
                 get_entropy = gini_gain,
                 num_stump_evaluations = 5,
                 max_depth = np.math.inf,
                 is_leaf = pure,
                 verbosity = 0,
                 label_encoder = None
                 ):
        self.verbosity = verbosity
        self.max_depth = max_depth
        self.get_distance_measure = distance_measure
        self.random_state = random_state
        self.is_leaf = is_leaf
        self.get_distance_measure = get_distance_measure
        self.setup_distance_measure = setup_distance_measure
        self.get_exemplars = get_exemplars
        self.label_encoder = label_encoder
        self.get_entropy = get_entropy
        self.num_stump_evaluations = num_stump_evaluations
        # below set in fit method
        self.depth = 0
        self.distance_measure = None
        self.stump = None
        self.branches = None
        self.classes_ = None

    def fit(self, X, y): # todo non-recursive version
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
                    label_encoder = self.label_encoder,
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
            if self.depth < self.max_depth and not self.is_leaf(y):
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
                        max_depth = self.max_depth,
                        label_encoder = self.label_encoder
                        )
                sub_tree.depth = self.depth + 1
            self.branches.append(sub_tree)
            if sub_tree is not None:
                sub_tree.fit(X, y)
        return self

    def predict_proba(self, X): # todo non-recursive version
        # todo checks
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
# todo constructor accept str name func / pointer

class PF(BaseClassifier):


    '''
    proximity forest classifier using an ensemble proximity trees.
    ----
    Parameters
    ----
    pick_exemplars_method : callable
        Method to pick exemplars from a set of instances and class labels
    param_pool : list of dicts
        a list of dictionaries containing a distance measure and corresponding parameter sources (distribution or
        predefined value)
    verbosity : int
        level of verbosity in output
    gain_method : callable
        a method to calculate the gain of splits / stumps in trees
    label_encoder : LabelEncoder
        a label encoder, can be pre-populated
    random_state : numpy RandomState
        a random state for sampling random numbers
    dimension : int
        dimension of the dataset to use. Defaults to zero for univariate datasets.
    num_stump_evaluations : int
        a tree parameter dictating the number of proximity stumps to produce at each node. Each stump has a random
        distance measure and distance measure parameter set. The stump with the best gain is used to split the data.
    num_trees : int
        the number of trees to construct
    is_leaf_method : callable
        a method which takes a split of data and produces a boolean value indicating whether the tree should
        continue splitting.
    ----
    Attributes
    ----
    trees : list of ProximityTrees
        ProximityTrees in this forest.
    label_encoder : LabelEncoder
        a label encoder, can be pre-populated
    classes_ :
        pointer to the label_encoder classes_
    '''

    __author__ = 'George Oastler (linkedin.com/goastler)'

    def __init__(self,
                 random_state = None,

                 num_trees = 100,
                 label_encoder = None,
                 distance_measure = None,
                 get_distance_measure = None,
                 get_exemplars = get_one_exemplar_per_class,
                 get_entropy = gini_gain,
                 verbosity = 0,
                 num_stump_evaluations = 5,
                 max_depth = np.math.inf,
                 is_leaf = pure,
                 setup_distance_measure = setup_distance_measure_getters,
                 ):
        self.is_leaf = is_leaf
        self.verbosity = verbosity
        self.max_depth = max_depth
        self.num_stump_evaluations = num_stump_evaluations
        self.get_exemplars = get_exemplars
        self.get_entropy = get_entropy
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
                print('tree ' + str(index) + ' building')
            tree = PT(
                    random_state = self.random_state,
                    verbosity = self.verbosity,
                    get_exemplars = self.get_exemplars,
                    get_entropy = self.get_entropy,
                    label_encoder = self.label_encoder,
                    distance_measure = self.distance_measure,
                    setup_distance_measure = self.setup_distance_measure,
                    get_distance_measure = self.get_distance_measure,
                    num_stump_evaluations = self.num_stump_evaluations,
                    max_depth = self.max_depth,
                    is_leaf = self.is_leaf,
                    )
            self.trees.append(tree)
            tree.fit(X, y)

    def predict_proba(self, X):
        # todo checks
        self.X = negative_dataframe_indices(X)
        predict_probas = np.zeros((X.shape[0], len(self.label_encoder.classes_)))
        count = 0
        for tree in self.trees:
            if self.verbosity > 0:
                print('tree ' + str(count) + ' predicting')
                count += 1
            tree_predict_probas = tree.predict_proba(X)
            predict_probas = np.add(predict_probas, tree_predict_probas)
        normalize(predict_probas, copy = False, norm = 'l1')
        return predict_probas
