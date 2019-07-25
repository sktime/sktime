# Proximity Forest: An effective and scalable distance-based classifier for time series
#
# author: George Oastler (linkedin.com/goastler; github.com/goastler)
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
# todo unit tests / sort out current unit tests
# todo logging package rather than print to screen
# todo get params avoid func pointer - use name
# todo set params use func name or func pointer
# todo constructor accept str name func / pointer
# todo duck-type functions
# todo comment-up transformers / util classes
# todo fix docstrings

__author__ = 'George Oastler (linkedin.com/goastler; github.com/goastler)'

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import check_random_state

from sktime.classifiers.base import BaseClassifier
from sktime.distances.elastic_cython import dtw_distance, erp_distance, lcss_distance, msm_distance, twe_distance, wdtw_distance
from sktime.transformers.series_to_series import CachedTransformer, DerivativeSlopeTransformer
from sktime.utils import comparison, dataset_properties
from sktime.utils.transformations import tabularise
from sktime.utils.validation import check_X, check_X_y


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
    """
    test whether a set of class labels are pure (i.e. all the same)
    ----
    Parameters
    ----
    y : 1d array like
        array of class labels
    ----
    Returns
    ----
    result : boolean
        whether the set of class labels is pure
    """
    # get unique class labels
    unique_class_labels = np.unique(np.array(y))
    # if more than 1 unique then not pure
    return len(unique_class_labels) <= 1


def gini_gain(y, y_subs):
    '''
    get gini score of a split, i.e. the gain from parent to children
    ----
    Parameters
    ----
    y : 1d array like
        array of class labels at parent
    y_subs : list of 1d array like
        list of array of class labels, one array per child
    ----
    Returns
    ----
    score : float
        gini score of the split from parent class labels to children. Note a higher score means better gain,
        i.e. a better split
    '''
    y = np.array(y)
    # find number of instances overall
    parent_n_instances = y.shape[0]
    # if parent has no instances then is pure
    if parent_n_instances == 0:
        for child in y_subs:
            if len(child) > 0:
                raise ValueError('children populated but parent empty')
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
            # weight score by proportion of instances at child compared to parent
            child_size = len(child_class_labels)
            child_score *= (child_size / parent_n_instances)
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
        gini score for the set of class labels (i.e. how pure they are). A larger score means more impurity. Zero means
        pure.
    '''
    y = np.array(y)
    # get number instances at node
    n_instances = y.shape[0]
    if n_instances > 0:
        # count each class
        unique_class_labels, class_counts = np.unique(y, return_counts = True)
        # subtract class entropy from current score for each class
        class_counts = np.divide(class_counts, n_instances)
        class_counts = np.power(class_counts, 2)
        sum = np.sum(class_counts)
        return 1 - sum
    else:
        # y is empty, therefore considered pure
        raise ValueError(' y empty')


def get_one_exemplar_per_class_proximity(proximity):
    '''
    unpack proximity object into X, y and random_state for picking exemplars.
    ----
    Parameters
    ----
    proximity : Proximity object
        Proximity like object containing the X, y and random_state variables required for picking exemplars.
    ----
    Returns
    ----
    chosen_instances : list
        list of the chosen exemplar instances.
    chosen_class_labels : array
        list of corresponding class labels for each of the chosen exemplar instances.
    '''
    return get_one_exemplar_per_class(proximity.X,
                                      proximity.y,
                                      proximity.random_state)


def get_one_exemplar_per_class(X, y, random_state):
    '''
    Pick one exemplar instance per class in the dataset.
    ----
    Parameters
    ----
    X : array-like or sparse matrix of shape = [n_samps, num_atts]
            The training input samples.  If a Pandas data frame is passed, the column _dim_to_use is extracted
    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        The class labels.
    random_state : numpy RandomState
        a random state for sampling random numbers
    ----
    Returns
    ----
    chosen_instances : list
        list of the chosen exemplar instances.
    chosen_class_labels : array
        list of corresponding class labels for each of the chosen exemplar instances.
    '''
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
    return {
            'distance_measure': [cython_wrapper(dtw_distance)],
            'w'               : stats.uniform(0, 0.25)
            }


def msm_distance_measure_getter(X):
    n_dimensions = 1  # todo use other dimensions
    return {
            'distance_measure': [cython_wrapper(msm_distance)],
            'dim_to_use'      : stats.randint(low = 0, high = n_dimensions),
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
    n_dimensions = 1  # todo use other dimensions
    return {
            'distance_measure': [cython_wrapper(erp_distance)],
            'dim_to_use'      : stats.randint(low = 0, high = n_dimensions),
            'g'               : stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp),
            'band_size'       : stats.randint(low = 0, high = max_raw_warping_window + 1)
            # scipy stats randint is exclusive on the max value, hence + 1
            }


def lcss_distance_measure_getter(X):
    stdp = dataset_properties.stdp(X)
    instance_length = dataset_properties.max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    n_dimensions = 1  # todo use other dimensions
    return {
            'distance_measure': [cython_wrapper(lcss_distance)],
            'dim_to_use'      : stats.randint(low = 0, high = n_dimensions),
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


def setup_all_distance_measure_getter(proximity):
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


def best_of_n_stumps(n):
    '''
    Generate the function to pick the best of n stump evaluations.
    ----
    Parameters
    ----
    n : int
        the number of stumps to evaluate before picking the best. Must be 1 or more.
    ----
    Returns
    ----
    find_best_stump : func
        function to find the best of n stumps.
    '''
    if n < 1:
        raise ValueError('n cannot be less than 1')

    def find_best_stump(proximity):
        '''
        Pick the best of n stump evaluations.
        ----
        Parameters
        ----
        proximity : Proximity like object
            the proximity object to split data from.
        ----
        Returns
        ----
        stump : ProximityStump
            the best stump / split of data of the n attempts.
        '''
        stumps = []
        # for n stumps
        for index in range(n):
            # duplicate tree configuration
            stump = ProximityStump(
                    random_state = proximity.random_state,
                    get_exemplars = proximity.get_exemplars,
                    distance_measure = proximity.distance_measure,
                    setup_distance_measure = proximity.setup_distance_measure,
                    get_distance_measure = proximity.get_distance_measure,
                    get_gain = proximity.get_gain,
                    verbosity = proximity.verbosity,
                    label_encoder = proximity.label_encoder,
                    n_jobs = proximity.n_jobs
                    )
            # grow the stump
            stump.fit(proximity.X, proximity.y)
            stump.grow()
            stumps.append(stump)
        # pick the best stump based upon gain
        stump = comparison.max(stumps, proximity.random_state, lambda stump: stump.entropy)
        return stump

    return find_best_stump


class ProximityStump(BaseClassifier):
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

    __author__ = 'George Oastler (linkedin.com/goastler; github.com/goastler)'

    def __init__(self,
                 random_state = None,
                 get_exemplars = get_one_exemplar_per_class_proximity,
                 setup_distance_measure = setup_all_distance_measure_getter,
                 get_distance_measure = None,
                 distance_measure = None,
                 get_gain = gini_gain,
                 verbosity = 0,
                 label_encoder = None,
                 n_jobs = 1,
                 ):
        self.setup_distance_measure = setup_distance_measure
        self.random_state = random_state
        self.get_distance_measure = get_distance_measure
        self.distance_measure = distance_measure
        self.pick_exemplars = get_exemplars
        self.get_gain = get_gain
        self.verbosity = verbosity
        self.label_encoder = label_encoder
        self.n_jobs = n_jobs
        # set in fit
        self.y_exemplar = None
        self.X_exemplar = None
        self.X_branches = None
        self.y_branches = None
        self.X = None
        self.y = None
        self.classes_ = None
        self.entropy = None

    @staticmethod
    def _distance_to_exemplars_inst(exemplars, instance, distance_measure):
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
        check_X(X)
        if self.n_jobs > 1 or self.n_jobs < 0:
            parallel = Parallel(self.n_jobs)
            distances = parallel(delayed(self._distance_to_exemplars_inst)
                                 (self.X_exemplar,
                                  X.iloc[index, :],
                                  self.distance_measure)
                                 for index in range(X.shape[0]))
        else:
            distances = [self._distance_to_exemplars_inst(self.X_exemplar,
                                                          X.iloc[index, :],
                                                          self.distance_measure)
                         for index in range(X.shape[0])]
        distances = np.vstack(np.array(distances))
        return distances

    def fit(self, X, y):
        check_X_y(X, y)
        self.X = dataset_properties.positive_dataframe_indices(X)
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
        check_X(X)
        n_instances = X.shape[0]
        distances = self.distance_to_exemplars(X)
        indices = np.empty(X.shape[0], dtype = int)
        for index in range(n_instances):
            exemplar_distances = distances[index]
            closest_exemplar_index = comparison.arg_min(exemplar_distances, self.random_state)
            indices[index] = closest_exemplar_index
        return indices

    def grow(self):
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
        check_X(X)
        X = dataset_properties.negative_dataframe_indices(X)
        distances = self.distance_to_exemplars(X)
        ones = np.ones(distances.shape)
        distances = np.add(distances, ones)
        distributions = np.divide(ones, distances)
        normalize(distributions, copy = False, norm = 'l1')
        return distributions


class ProximityTree(BaseClassifier):
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

    __author__ = 'George Oastler (linkedin.com/goastler; github.com/goastler)'

    def __init__(self,
                 # note: any changes of these params must be reflected in the fit method for building trees / clones
                 random_state = None,
                 get_exemplars = get_one_exemplar_per_class_proximity,
                 distance_measure = None,
                 get_distance_measure = None,
                 setup_distance_measure = setup_all_distance_measure_getter,
                 get_gain = gini_gain,
                 max_depth = np.math.inf,
                 is_leaf = pure,
                 verbosity = 0,
                 label_encoder = None,
                 n_jobs = 1,
                 find_stump = best_of_n_stumps(5),
                 ):
        self.verbosity = verbosity
        self.find_stump = find_stump
        self.max_depth = max_depth
        self.get_distance_measure = distance_measure
        self.random_state = random_state
        self.is_leaf = is_leaf
        self.get_distance_measure = get_distance_measure
        self.setup_distance_measure = setup_distance_measure
        self.get_exemplars = get_exemplars
        self.label_encoder = label_encoder
        self.get_gain = get_gain
        self.n_jobs = n_jobs
        self.depth = 0
        # below set in fit method
        self.distance_measure = None
        self.stump = None
        self.branches = None
        self.X = None
        self.y = None
        self.classes_ = None

    def fit(self, X, y):
        check_X_y(X, y)
        self.X = dataset_properties.positive_dataframe_indices(X)
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
        self.stump = self.find_stump(self)
        n_branches = len(self.stump.y_exemplar)
        self.branches = [None] * n_branches
        if self.depth < self.max_depth:
            for index in range(n_branches):
                sub_y = self.stump.y_branches[index]
                if not self.is_leaf(sub_y):
                    sub_tree = ProximityTree(
                            random_state = self.random_state,
                            get_exemplars = self.get_exemplars,
                            distance_measure = self.distance_measure,
                            setup_distance_measure = self.setup_distance_measure,
                            get_distance_measure = self.get_distance_measure,
                            get_gain = self.get_gain,
                            is_leaf = self.is_leaf,
                            verbosity = self.verbosity,
                            max_depth = self.max_depth,
                            label_encoder = self.label_encoder,
                            n_jobs = self.n_jobs
                            )
                    sub_tree.depth = self.depth + 1
                    self.branches[index] = sub_tree
                    sub_X = self.stump.X_branches[index]
                    sub_tree.fit(sub_X, sub_y)
        return self

    def predict_proba(self, X):
        check_X(X)
        X = dataset_properties.negative_dataframe_indices(X)
        closest_exemplar_indices = self.stump.find_closest_exemplar_indices(X)
        n_classes = len(self.label_encoder.classes_)
        distribution = np.zeros((X.shape[0], n_classes))
        for index in range(len(self.branches)):
            indices = np.argwhere(closest_exemplar_indices == index)
            if indices.shape[0] > 0:
                indices = np.ravel(indices)
                sub_tree = self.branches[index]
                if sub_tree is None:
                    sub_distribution = np.zeros(n_classes)
                    class_label = self.stump.y_exemplar[index]
                    sub_distribution[class_label] = 1
                else:
                    sub_X = X.iloc[indices, :]
                    sub_distribution = sub_tree.predict_proba(sub_X)
                np.add.at(distribution, indices, sub_distribution)
        normalize(distribution, copy = False, norm = 'l1')
        return distribution


class ProximityForest(BaseClassifier):
    """
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
    """

    __author__ = 'George Oastler (linkedin.com/goastler; github.com/goastler)'

    def __init__(self,
                 random_state = None,
                 n_trees = 100,
                 label_encoder = None,
                 distance_measure = None,
                 get_distance_measure = None,
                 get_exemplars = get_one_exemplar_per_class_proximity,
                 get_gain = gini_gain,
                 verbosity = 0,
                 max_depth = np.math.inf,
                 is_leaf = pure,
                 n_jobs = 1,
                 setup_distance_measure_getter = setup_all_distance_measure_getter,
                 ):
        self.is_leaf = is_leaf
        self.verbosity = verbosity
        self.max_depth = max_depth
        self.get_exemplars = get_exemplars
        self.get_gain = get_gain
        self.random_state = random_state
        self.n_trees = n_trees
        self.n_jobs = n_jobs
        self.label_encoder = label_encoder
        self.get_distance_measure = get_distance_measure
        self.setup_distance_measure_getter = setup_distance_measure_getter
        self.distance_measure = distance_measure
        # set in fit method
        self.trees = None
        self.X = None
        self.y = None
        self.classes_ = None

    def _fit_tree(self, X, y, index, random_state):
        if self.verbosity > 0:
            print('tree ' + str(index) + ' building')
        tree = ProximityTree(
                random_state = random_state,
                verbosity = self.verbosity,
                get_exemplars = self.get_exemplars,
                get_gain = self.get_gain,
                label_encoder = self.label_encoder,
                distance_measure = self.distance_measure,
                setup_distance_measure = self.setup_distance_measure_getter,
                get_distance_measure = self.get_distance_measure,
                max_depth = self.max_depth,
                is_leaf = self.is_leaf,
                n_jobs = 1,
                )
        tree.fit(X, y)
        return tree

    def fit(self, X, y):
        check_X_y(X, y)
        self.X = dataset_properties.positive_dataframe_indices(X)
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
                self.get_distance_measure = self.setup_distance_measure_getter(self)
            self.distance_measure = self.get_distance_measure(self)
        if self.n_jobs > 1 or self.n_jobs < 0:
            parallel = Parallel(self.n_jobs)
            self.trees = parallel(delayed(self._fit_tree)(X, y, index, self.random_state.randint(0, self.n_trees))
                                  for index in range(self.n_trees))
        else:
            self.trees = [self._fit_tree(X, y, index, self.random_state.randint(0, self.n_trees))
                          for index in range(self.n_trees)]
        return self

    @staticmethod
    def _predict_proba_tree(X, tree):
        return tree.predict_proba(X)

    def predict_proba(self, X):
        check_X(X)
        X = dataset_properties.negative_dataframe_indices(X)
        if self.n_jobs > 1 or self.n_jobs < 0:
            parallel = Parallel(self.n_jobs)
            distributions = parallel(delayed(self._predict_proba_tree)(X, tree) for tree in self.trees)
        else:
            distributions = [self._predict_proba_tree(X, tree) for tree in self.trees]
        distributions = np.array(distributions)
        distributions = np.sum(distributions, axis = 0)
        normalize(distributions, copy = False, norm = 'l1')
        return distributions
