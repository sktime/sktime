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
# todo expand debug printing capability

__author__ = "George Oastler"

import numpy as np
from numpy.ma import floor
from pandas import DataFrame, Series
from scipy.stats import randint, uniform
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import check_random_state

from .base import BaseClassifier
from ..distances.elastic_cython import (
    ddtw_distance, dtw_distance, erp_distance, lcss_distance, msm_distance, wddtw_distance, wdtw_distance,
    )
from ..utils import comparison
from ..utils import dataset_properties
from ..utils.transformations import tabularise
from ..utils.validation import check_X_y


def get_default_dimension():
    """
    returns default dimension to use in a dataset. Defaults to 0 for univariate datasets.
    ----
    Returns
    ----
    result : int
        default dimension of a dataset to use
    """
    return 0


def get_default_num_trees():
    """
    returns default number of trees to make in a proximity forest
    ----
    Returns
    ----
    result : int
        default number of trees
    """
    return 100


def get_default_gain_method():
    """
    returns default gain method for a split at a tree node
    ----
    Returns
    ----
    result : callable
        default gain method
    """
    return gini


def get_default_num_stump_evaluations():
    """
    returns default r (number of splits) to try at a tree node
    ----
    Returns
    ----
    result : int
        default number of splits to examine
    """
    return 5


def get_default_is_leaf_method():
    """
    returns default method for checking whether a tree should branch further or not
    ----
    Returns
    ----
    result : callable
        default method to check whether a tree node is a leaf or not
    """
    return pure


def get_default_pick_exemplars_method():
    """
    returns default method for picking exemplar instances from a dataset
    ----
    Returns
    ----
    result : callable
        default method to pick exemplars from a dataset (set of instances and class labels)
    """
    return pick_one_exemplar_per_class


def pure(y):
    """
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
    """
    # get unique class labels
    unique_class_labels = np.unique(y)
    # if more than 1 unique then not pure
    return len(unique_class_labels) <= 1


def gini(parent_class_labels, children_class_labels):
    """
    get gini score of a split, i.e. the gain from parent to children
    ----
    Parameters
    ----
    parent_class_labels : 1d numpy array
        array of class labels at parent
    children_class_labels : list of 1d numpy array
        list of array of class labels, one array per child
    ----
    Returns
    ----
    score : float
        gini score of the split from parent class labels to children. Note the gini score is scaled to be between 0
        and 1. 1 == pure, 0 == not pure
    """
    # find gini for parent node
    parent_score = gini_node(parent_class_labels)
    # find number of instances overall
    parent_num_instances = parent_class_labels.shape[0]
    # sum the children's gini scores
    children_score_sum = 0
    for index in range(0, len(children_class_labels)):
        child_class_labels = children_class_labels[index]
        # find gini score for this child
        child_score = gini_node(child_class_labels)
        # weight score by proportion of instances at child compared to parent
        child_size = len(child_class_labels)
        child_score *= (child_size / parent_num_instances)
        # add to cumulative sum
        children_score_sum += child_score
    # gini outputs relative improvement
    score = parent_score - children_score_sum
    return score


def gini_node(y):
    """
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
    """
    # get number instances at node
    num_instances = y.shape[0]
    score = 1
    if num_instances > 0:
        # count each class
        unique_class_labels, class_counts = np.unique(y, return_counts = True)
        # subtract class entropy from current score for each class
        for index in range(0, len(unique_class_labels)):
            class_count = class_counts[index]
            proportion = class_count / num_instances
            sq_proportion = np.math.pow(proportion, 2)
            score -= sq_proportion
    # double score as gini is between 0 and 0.5, we need 0 and 1
    score *= 2
    return score


# todo info gain
def information_gain(parent_class_labels, children_class_labels):
    raise NotImplementedError()


# todo chi sq
def chi_squared(parent_class_labels, children_class_labels):
    raise NotImplementedError()


def pick_one_exemplar_per_class(X, y, random_state):
    """
    pick one random exemplar instance per class
    ----
    Parameters
    ----
    X : panda dataframe
        instances representing a dataset
    y : 1d numpy array
        array of class labels, one for each instance in the instances panda dataframe parameter
    ----
    Returns
    ----
    chosen_instances : panda dataframe
        the chosen exemplar instances
    chosen_class_labels : numpy 1d array
        array of class labels corresponding to the exemplar instances
        the exemplar instances class labels
    remaining_instances : panda dataframe
        the remaining instances after exemplars have been removed
    remaining_class_labels : numpy 1d array
        array of class labels corresponding to the exemplar instances
        the remaining instances class labels after picking exemplars
    """
    # find unique class labels
    unique_class_labels = np.unique(y)
    num_unique_class_labels = len(unique_class_labels)
    chosen_instances = []
    chosen_class_labels = np.empty(num_unique_class_labels, dtype = int)
    chosen_indices = np.empty(num_unique_class_labels, dtype = int)
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
        chosen_instances.append(instance)
        chosen_class_labels[class_label_index] = class_label
        chosen_indices[class_label_index] = index
    # remove exemplar class labels from dataset - note this returns a copy, not inplace!
    remaining_class_labels = np.delete(y, chosen_indices)
    # remove exemplar instances from dataset - note this returns a copy, not inplace!
    remaining_instances = X.drop(X.index[chosen_indices])
    return chosen_instances, chosen_class_labels, remaining_instances, remaining_class_labels


def get_all_distance_measures_param_pool(X, dimension):
    """
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
    """
    # find dataset properties
    instance_length = dataset_properties.max_instance_length(X,
                                                             dimension)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = floor((instance_length + 1) / 4)
    max_warping_window_percentage = max_raw_warping_window / instance_length
    stdp = dataset_properties.stdp(X)
    # setup param pool dictionary array (same structure as sklearn's GridSearchCV params!)
    param_pool = [
            {
                    ProximityStump.get_distance_measure_key(): [dtw_distance],
                    'w'                                      : uniform(0, max_warping_window_percentage)
                    },
            {
                    ProximityStump.get_distance_measure_key(): [ddtw_distance],
                    'w'                                      : uniform(0, max_warping_window_percentage)
                    },
            {
                    ProximityStump.get_distance_measure_key(): [wdtw_distance],
                    'g'                                      : uniform(0, 1)
                    },
            {
                    ProximityStump.get_distance_measure_key(): [wddtw_distance],
                    'g'                                      : uniform(0, 1)
                    },
            {
                    ProximityStump.get_distance_measure_key(): [lcss_distance],
                    'epsilon'                                : uniform(0.2 * stdp, stdp),
                    'delta'                                  : randint(low = 0, high = max_raw_warping_window)
                    },
            {
                    ProximityStump.get_distance_measure_key(): [erp_distance],
                    'g'                                      : uniform(0.2 * stdp, 0.8 * stdp),
                    'band_size'                              : randint(low = 0, high = max_raw_warping_window)
                    },
            # {Split.get_distance_measure_key(): [twe_distance],
            #  'g': uniform(0.2 * stdp, 0.8 * stdp),
            #  'band_size': randint(low=0, high=max_raw_warping_window)},
            {
                    ProximityStump.get_distance_measure_key(): [msm_distance],
                    'c'                                      : [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325,
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
                    },
            ]
    return param_pool

def get_default_param_perm(X, dimension):
    """
    get default parameter permutation, i.e. euclidean distance
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
    param_perm : dict
        a dictionary of a distance measure (dtw) and corresponding parameters (window size)
    """
    # find dataset properties
    instance_length = dataset_properties.max_instance_length(X,
                                                             dimension)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = floor((instance_length + 1) / 4)
    return {
            ProximityStump.get_distance_measure_key(): dtw_distance,
            'w'                                      : max_raw_warping_window
            }


class ProximityStump(BaseClassifier):
    """
    proximity tree classifier of depth 1 - in other words, a k=1 nearest neighbour classifier with neighbourhood limited
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
    debug : boolean
        whether to print debug info
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
    """

    __author__ = 'George Oastler (linkedin.com/goastler)'

    def __init__(self,
                 pick_exemplars_method = get_default_pick_exemplars_method(),
                 param_perm = get_default_param_perm,
                 gain_method = get_default_gain_method(),
                 label_encoder = None,
                 dimension = get_default_dimension(),
                 random_state = None,
                 debug = False):
        super().__init__()
        self.random_state = random_state
        self.param_perm = param_perm
        self.dimension = dimension
        self.gain_method = gain_method
        self.pick_exemplars_method = pick_exemplars_method
        # vars set in the fit method
        self.exemplar_instances = None
        self.exemplar_class_labels = None
        self.remaining_instances = None
        self.remaining_class_labels = None
        self.branch_instances = None
        self.branch_class_labels = None
        self.distance_measure_param_perm = None
        self.distance_measure = None
        self.gain = None
        self.debug = debug
        self.label_encoder = label_encoder
        self.classes_ = None

    @staticmethod
    def get_distance_measure_key():
        """
        get the key for the distance measure. This key is required for picking the distance measure out of the
        param_perm constructor parameter.
        ----
        Returns
        ----
        key : string
            key for the distance measure for the param_perm dict
        """
        return 'dm'

    def fit(self, X, y, input_checks = True):
        """
        model a dataset using this proximity stump
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts]
            The training input samples.  If a Pandas data frame is passed, the column _dim_to_use is extracted
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The class labels.

        Returns
        -------
        self : object
        """
        # checks
        if input_checks:
            check_X_y(X, y)
        if callable(self.param_perm):
            self.param_perm = self.param_perm(X, self.dimension)
        if not isinstance(self.param_perm, dict):
            raise ValueError("parameter permutation must be a dict or callable to obtain dict")
        if not callable(self.gain_method):
            raise ValueError("gain method must be callable")
        if not callable(self.pick_exemplars_method):
            raise ValueError("gain method must be callable")
        self.random_state = check_random_state(self.random_state)
        # if label encoder not setup, make a new one and train it
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
            y = self.label_encoder.transform(y)
        # if distance measure not extracted from parameter permutation
        if self.distance_measure is None:
            key = self.get_distance_measure_key()  # get the key for the distance measure var in the param perm dict
            self.distance_measure = self.param_perm[key]
            # copy so not available to outside world
            self.distance_measure_param_perm = self.param_perm.copy()
            # delete as we don't want to pass the distance measure as a parameter to itself!
            del self.distance_measure_param_perm[key]
        self.classes_ = self.label_encoder.classes_
        # get exemplars from dataset
        self.exemplar_instances, self.exemplar_class_labels, self.remaining_instances, self.remaining_class_labels = \
            self.pick_exemplars_method(X, y, self.random_state)
        # find distances of remaining instances to the exemplars
        distances = self.exemplar_distances(self.remaining_instances)
        num_exemplars = len(self.exemplar_instances)
        self.branch_class_labels = []
        self.branch_instances = []
        # for each branch add a list for the instances and class labels closest to the exemplar instance for that branch
        for index in range(0, num_exemplars):
            self.branch_instances.append([])
            self.branch_class_labels.append([])
        num_instances = self.remaining_instances.shape[0]
        # for each instance
        for instance_index in range(0, num_instances):
            # find the distance to each exemplar
            exemplar_distances = distances[instance_index]
            instance = self.remaining_instances.iloc[instance_index, :]
            class_label = self.remaining_class_labels[instance_index]
            # pick the closest exemplar (min distance)
            closest_exemplar_index = comparison.arg_min(exemplar_distances, self.random_state)
            # add the instance to the corresponding list for the exemplar branch
            self.branch_instances[closest_exemplar_index].append(instance)
            self.branch_class_labels[closest_exemplar_index].append(class_label)
        # convert lists to panda dataframe and numpy array for ease of use in other things (e.g. in a tree where
        # branched instances / class labels are used in the next level
        for index in range(0, num_exemplars):
            self.branch_class_labels[index] = np.array(self.branch_class_labels[index])
            self.branch_instances[index] = DataFrame(self.branch_instances[index])
        # work out the gain for this split / stump
        self.gain = self.gain_method(y, self.branch_class_labels)
        return self

    def exemplar_distances(self, X, input_checks = True):
        """
        find the distance from the given instances to each exemplar instance
        ----
        Parameters
        ----
        X : panda dataframe
            instances of the dataset
        input_checks : boolean
            whether to verify the dataset (e.g. dimensions, etc)
        ----
        Returns
        ----
        distances : 2d list
            list of distance corresponding to each exemplar instance (instances by distances)
        """
        # check data
        if input_checks:
            check_X_y(X)
        num_instances = X.shape[0]
        distances = []
        # for each instance
        for instance_index in range(0, num_instances):
            # find the distances to each exemplar
            instance = X.iloc[instance_index, :]
            distances_inst = self.exemplar_distance_inst(instance, input_checks = False)
            # add distances to the list (at the corresponding index to the instance being tested)
            distances.append(distances_inst)
        return distances

    def exemplar_distance_inst(self, instance, input_checks = True):
        """
        find the distance from the given instance to each exemplar instance. Note this returns distance + 1 for
        efficiency in other methods.
        ----
        Parameters
        ----
        instance : panda dataframe
            instance of the dataset
        input_checks : boolean
            whether to verify the dataset (e.g. dimensions, etc)
        ----
        Returns
        ----
        distances : list
            list of distance corresponding to each exemplar instance
        """
        # check data
        if input_checks:
            if not isinstance(instance, Series):
                raise ValueError("instance not a panda series")
        num_exemplars = len(self.exemplar_instances)
        distances = []
        # for each exemplar
        for exemplar_index in range(0, num_exemplars):
            # find the distance to the given instance
            exemplar = self.exemplar_instances[exemplar_index]
            distance = self._find_distance(exemplar, instance)
            # increment distance so at least 1
            distance += 1
            # add it to the list (at same index as exemplar instance index)
            distances.append(distance)
        return distances

    def predict_proba(self, X, input_checks = True):
        """
        classify instances
        ----
        Parameters
        ----
        X : panda dataframe
            instances of the dataset
        input_checks : boolean
            whether to verify the dataset (e.g. dimensions, etc)
        ----
        Returns
        ----
        predictions : 2d numpy array (instance by class)
            array of prediction arrays. Each array has <num classes> values reflecting probability of each class.
        """
        # check data
        if input_checks:
            check_X_y(X)
        # find distances to each exemplar for each test instance
        distances = self.exemplar_distances(X, input_checks = False)
        distances = np.array(distances)
        ones = np.ones(distances.shape)
        distributions = np.divide(ones, distances)
        normalize(distributions, copy = False, norm = 'l1')
        return distributions

    def _find_distance(self, instance_a, instance_b, input_checks = True):
        """
        find distance between two instances using distance measure + distance measure parameters
        ----
        Parameters
        ----
        instance_a : panda dataframe
            instance of the dataset
        instance_a : panda dataframe
            another instance of the dataset
        input_checks : boolean
            whether to verify the dataset (e.g. dimensions, etc)
        ----
        Returns
        ----
        distance : float
            value indicating how similar the two instances are
        """
        if input_checks:
            if not isinstance(instance_a, Series):
                raise ValueError("instance not a panda series")
            if not isinstance(instance_b, Series):
                raise ValueError("instance not a panda series")
        # flatten both instances and transpose for cython parameter format
        instance_a = tabularise(instance_a, return_array = True)
        instance_b = tabularise(instance_b, return_array = True)
        instance_a = np.transpose(instance_a)
        instance_b = np.transpose(instance_b)
        # find distance
        params = self.distance_measure_param_perm
        # if distance measure uses dimension
        if self.distance_measure == msm_distance or self.distance_measure == lcss_distance or self.distance_measure \
                == erp_distance:
            # copy the parameters
            params = params.copy()
            # add the dimension to use
            params['dim_to_use'] = self.dimension
        return self.distance_measure(instance_a, instance_b, **params)


class ProximityTree(BaseClassifier):
    """
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
    debug : boolean
        whether to print debug info
    num_stump_evaluations : int
        the number of proximity stumps to produce at each node. Each stump has a random distance measure and distance
        measure parameter set. The stump with the best gain is used to split the data.
    is_leaf_method : callable
        a method which takes a split of data and produces a boolean value indicating whether the tree should continue
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
    """

    __author__ = 'George Oastler (linkedin.com/goastler)'

    def __init__(self,
                 gain_method = get_default_gain_method(),
                 num_stump_evaluations = get_default_num_stump_evaluations(),
                 max_depth = np.math.inf,
                 dimension = get_default_dimension(),
                 random_state = None,
                 debug = False,
                 is_leaf_method = get_default_is_leaf_method(),
                 label_encoder = None,
                 pick_exemplars_method = get_default_pick_exemplars_method(),
                 param_pool = get_all_distance_measures_param_pool):
        super().__init__()
        self.random_state = random_state
        self.gain_method = gain_method
        self.num_stump_evaluations = num_stump_evaluations
        self.max_depth = max_depth
        self.label_encoder = label_encoder
        self.pick_exemplars_method = pick_exemplars_method
        self.is_leaf_method = is_leaf_method
        self.param_pool = param_pool
        self.dimension = dimension
        self.level = 0
        self.debug = debug
        # vars set in the fit method
        self.branches = None
        self.stump = None
        self.classes_ = None

    def predict_proba(self, X, input_checks = True):
        """
        classify instances
        ----
        Parameters
        ----
        instances : panda dataframe
            instances of the dataset
        input_checks : boolean
            whether to verify the dataset (e.g. dimensions, etc)
        ----
        Returns
        ----
        predictions : 2d numpy array (instance by class)
            array of prediction arrays. Each array has <num classes> values reflecting probability of each
            class.
        """
        # check data
        if input_checks:
            check_X_y(X)
        num_instances = X.shape[0]
        distributions = []
        # for each instance
        for instance_index in range(0, num_instances):
            instance = X.iloc[instance_index, :]
            previous_tree = None
            tree = self
            closest_exemplar_index = -1
            # traverse the tree
            while tree:
                # find the distances to each exemplar
                distances = tree.stump.exemplar_distance_inst(instance)
                # find closest exemplar
                closest_exemplar_index = comparison.arg_min(distances, tree.random_state)
                # move to the tree corresponding to the closest exemplar
                previous_tree = tree
                tree = tree.branches[closest_exemplar_index]
                # if the tree is none then it is a leaf node
            # jump back to the previous tree (one before none)
            tree = previous_tree
            # get the class label for the closest exemplar at this node
            prediction = [0] * len(self.label_encoder.classes_)
            closest_exemplar_class_label = tree.stump.exemplar_class_labels[closest_exemplar_index]
            # increment the prediction at the closest exemplar's class label index
            prediction[closest_exemplar_class_label] += 1
            # add to predictions
            distributions.append(prediction)
        # normalise the predictions
        distributions = np.array(distributions)
        normalize(distributions, copy = False, norm = 'l1')
        return distributions

    def _branch(self, X, y):
        """
        branch into further trees based upon proximity found in stump
        ----
        Parameters
        ----
        X : panda dataframe
            instances of the dataset
        y : numpy 1d array
            class labels corresponding to each instance
        """
        # find best stump (split of data)
        self.stump = self._get_best_stump(X, y)
        num_branches = len(self.stump.branch_instances)
        self.branches = []
        # providing max depth not exceeded
        if self.level < self.max_depth:
            # for each branch (each exemplar instance)
            for branch_index in range(0, num_branches):
                # find class label for this branch, i.e. the class label of the exemplar instance
                branch_class_labels = self.stump.branch_class_labels[branch_index]
                # if not a leaf node
                if not self.is_leaf_method(branch_class_labels):
                    # construct a new tree (cloning parameters of this tree) to use on the branch's instances
                    tree = ProximityTree(
                            gain_method = self.gain_method,
                            num_stump_evaluations = self.num_stump_evaluations,
                            random_state = self.random_state,
                            is_leaf_method = self.is_leaf_method,
                            max_depth = self.max_depth,
                            label_encoder = self.label_encoder,
                            param_pool = self.param_pool,
                            dimension = self.dimension,
                            )
                    # increment the level
                    tree.level = self.level + 1
                    # add tree to branches list
                    self.branches.append(tree)
                else:
                    # add none to branches list indicating a leaf node
                    self.branches.append(None)

    def fit(self, X, y, input_checks = True):
        """
        model a dataset using this proximity tree
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts]
            The training input samples.  If a Pandas data frame is passed, the column _dim_to_use is extracted
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The class labels.

        Returns
        -------
        self : object
        """
        # check data
        if input_checks:
            check_X_y(X, y)
        # check parameter values
        if self.max_depth < 0:
            raise ValueError('max depth cannot be less than 0')
        if self.num_stump_evaluations < 1:
            raise ValueError('r cannot be less than 1')
        if not callable(self.gain_method):
            raise RuntimeError('gain method not callable')
        if not callable(self.pick_exemplars_method):
            raise RuntimeError('pick exemplars method not callable')
        if not callable(self.is_leaf_method):
            raise RuntimeError('is leaf method not callable')
        # if param_pool is obtained using train instances
        if callable(self.param_pool):
            # call param_pool function giving train instances as parameter
            self.param_pool = self.param_pool(X, self.dimension)
        self.random_state = check_random_state(self.random_state)
        # train label encoder if not already
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
            y = self.label_encoder.transform(y)
        self.classes_ = self.label_encoder.classes_
        # train the tree using a stacking top down decision tree induction algorithm. This constructs the tree in a
        # iterative breadth first manner
        # 3 queues, one for trees, one for instances, one for class labels. Indices correspond, e.g. tree_queue[4] is
        # a tree which train on the instances at instances_queue[4] and said instance have class labels at
        # class_labels_queue[4]
        # add this tree to the queue with the full dataset and class labels
        tree_queue = [self]
        instances_queue = [X]
        class_labels_queue = [y]
        # while the queue is not empty
        while tree_queue:
            # get the next tree, instances and class labels in the queue
            tree = tree_queue.pop()
            instances = instances_queue.pop()
            class_labels = class_labels_queue.pop()
            # branch the tree
            tree._branch(instances, class_labels)
            # for each branch
            for branch_index in range(0, len(tree.branches)):
                # get the sub tree for that branch
                sub_tree = tree.branches[branch_index]
                # if it is none then it is a leaf, i.e. do nothing
                if sub_tree is not None:
                    # otherwise add the sub tree to the tree queue for further branching
                    tree_queue.insert(0, sub_tree)
                    instances = tree.stump.branch_instances[branch_index]
                    class_labels = tree.stump.branch_class_labels[branch_index]
                    instances_queue.insert(0, instances)
                    class_labels_queue.insert(0, class_labels)
        # queue empty so tree has branched into sub tree until contain only leaf nodes
        return self

    def _get_rand_param_perm(self, params = None):
        """
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
        """
        #
        if params is None:
            params = self.param_pool
        param_pool = self.random_state.choice(params)
        permutation = self._pick_param_permutation(param_pool)
        return permutation

    def _get_best_stump(self, X, y):
        stumps = np.empty(self.num_stump_evaluations, dtype = object)
        for index in range(0, self.num_stump_evaluations):
            split = self._pick_rand_stump(X, y)
            stumps[index] = split
        best_stump = comparison.best(stumps, lambda a, b: a.gain - b.gain, self.random_state)
        return best_stump

    def _pick_rand_stump(self, X, y):
        param_perm = self._get_rand_param_perm()
        stump = ProximityStump(pick_exemplars_method = self.pick_exemplars_method,
                               random_state = self.random_state,
                               gain_method = self.gain_method,
                               label_encoder = self.label_encoder,
                               param_perm = param_perm)
        stump.fit(X, y, input_checks = False)
        return stump

    def _pick_param_permutation(self, param_pool):
        """
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
        """
        # construct empty permutation
        param_perm = {}
        # for each parameter
        for param_name, param_values in param_pool.items():
            # if it is a list
            if isinstance(param_values, list):
                # randomly pick a value
                param_value = self.random_state.choice(param_values)
                # if the value is another dict then get a random parameter permutation from that dict (recursive over
                # 2 funcs)
                if isinstance(param_value, dict):
                    param_value = self._get_rand_param_perm(param_value)
            # else if parameter is a distribution
            elif hasattr(param_values, 'rvs'):
                # sample from the distribution
                param_value = param_values.rvs(random_state = self.random_state)
            else:
                # otherwise we don't know how to obtain a value from the parameter
                raise Exception('unknown type of parameter pool')
            # add parameter name and value to permutation
            param_perm[param_name] = param_value
        return param_perm


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
    debug : boolean
        whether to print debug info
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
    debug : boolean
        whether to print debug info
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

    __author__ = 'George Oastler (linkedin.com/goastler)'

    def __init__(self,
                 pick_exemplars_method = get_default_pick_exemplars_method(),
                 gain_method = get_default_gain_method(),
                 num_stump_evaluations = get_default_num_stump_evaluations(),
                 dimension = get_default_dimension(),
                 num_trees = get_default_num_trees(),
                 random_state = None,
                 is_leaf_method = get_default_is_leaf_method(),
                 max_depth = np.math.inf,
                 label_encoder = None,
                 debug = False,
                 param_pool = get_all_distance_measures_param_pool):
        super().__init__()
        self.random_state = random_state
        self.gain_method = gain_method
        self.num_stump_evaluations = num_stump_evaluations
        self.label_encoder = label_encoder
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.debug = debug
        self.dimension = dimension
        self.is_leaf_method = is_leaf_method
        self.pick_exemplars_method = pick_exemplars_method
        self.param_pool = param_pool
        # below set in fit method
        self.trees = None
        self.classes_ = None

    def fit(self, X, y, input_checks = True):
        """
        model a dataset using this proximity forest
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts]
            The training input samples.  If a Pandas data frame is passed, the column _dim_to_use is extracted
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The class labels.

        Returns
        -------
        self : object
        """
        # check data
        if input_checks:
            check_X_y(X, y)
        # check parameter values
        if self.num_trees < 1:
            raise ValueError('number of trees cannot be less than 1')
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
            y = self.label_encoder.transform(y)
        if callable(self.param_pool):
            # if param pool obtained via train instances then call it
            self.param_pool = self.param_pool(X, self.dimension)
        self.random_state = check_random_state(self.random_state)
        self.classes_ = self.label_encoder.classes_
        # init list of trees
        self.trees = []
        # for each tree
        for tree_index in range(0, self.num_trees):
            if self.debug:
                print("constructing tree " + str(tree_index))
            # build tree from forest parameters
            tree = ProximityTree(
                    gain_method = self.gain_method,
                    num_stump_evaluations = self.num_stump_evaluations,
                    random_state = self.random_state,
                    is_leaf_method = self.is_leaf_method,
                    max_depth = self.max_depth,
                    label_encoder = self.label_encoder,
                    param_pool = self.param_pool,
                    pick_exemplars_method = self.pick_exemplars_method,
                    dimension = self.dimension,  # todo could randomise?
                    )
            # build tree on dataset
            tree.fit(X, y, input_checks = False)
            # append tree to tree list
            self.trees.append(tree)
        return self

    def predict_proba(self, X, input_checks = True):
        """
        classify instances
        ----
        Parameters
        ----
        X : panda dataframe
            instances of the dataset
        input_checks : boolean
            whether to verify the dataset (e.g. dimensions, etc)
        ----
        Returns
        ----
        predictions : 2d numpy array (instance by class)
            array of prediction arrays. Each array has <num classes> values reflecting probability of each
            class.
        """
        # check data
        if input_checks:
            check_X_y(X)
        # store sum of overall predictions. (majority vote)
        overall_predict_probas = np.zeros((X.shape[0], len(self.label_encoder.classes_)))
        # for each tree
        for tree in self.trees:
            # add the tree's predictions to the overall
            predict_probas = tree.predict_proba(X, input_checks = False)
            overall_predict_probas = np.add(overall_predict_probas, predict_probas)
        # normalise the overall predictions
        normalize(overall_predict_probas, copy = False, norm = 'l1')
        return overall_predict_probas
