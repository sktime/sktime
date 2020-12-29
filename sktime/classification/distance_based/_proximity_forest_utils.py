# -*- coding: utf-8 -*-
# find the index of the best value in the array
import numpy as np


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
def pick_from_indices(array, indices):
    picked = []
    for index in indices:
        picked.append(array[index])
    return picked


# find best values in array
def bests(array, comparator):
    indices = arg_bests(array, comparator)
    return pick_from_indices(array, indices)


# find min values in array
def mins(array, getter=None):
    indices = arg_mins(array, getter)
    return pick_from_indices(array, indices)


# find max values in array
def maxs(array, getter=None):
    indices = arg_maxs(array, getter)
    return pick_from_indices(array, indices)


# find min value in array, randomly breaking ties
def min(array, rand, getter=None):
    index = arg_min(array, rand, getter)
    return array[index]


# find index of max value in array, randomly breaking ties
def arg_max(array, rand, getter=None):
    return rand.choice(arg_maxs(array, getter))


# find max value in array, randomly breaking ties
def max(array, rand, getter=None):
    index = arg_max(array, rand, getter)
    return array[index]


# find best value in array, randomly breaking ties
def best(array, comparator, rand):
    return rand.choice(bests(array, comparator))


# find index of best value in array, randomly breaking ties
def arg_best(array, comparator, rand):
    return rand.choice(arg_bests(array, comparator))


# find index of min value in array, randomly breaking ties
def arg_min(array, rand, getter=None):
    return rand.choice(arg_mins(array, getter))


# find indices of best value in array, randomly breaking ties
def arg_mins(array, getter=None):
    return arg_bests(array, chain(less_than, getter))


# find indices of max value in array, randomly breaking ties
def arg_maxs(array, getter=None):
    return arg_bests(array, chain(more_than, getter))


# obtain a value before using in func
def chain(func, getter=None):
    if getter is None:
        return func
    else:
        return lambda a, b: func(getter(a), getter(b))


# test if a is more than b
def more_than(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


# test if a is less than b
def less_than(a, b):
    if a < b:
        return 1
    elif a > b:
        return -1
    else:
        return 0


def negative_dataframe_indices(X):
    if np.any(X.index >= 0) or len(np.unique(X.index)) > 1:
        X = X.copy(deep=True)
        X.index = np.arange(-1, -len(X.index) - 1, step=-1)
    return X


def positive_dataframe_indices(X):
    if np.any(X.index < 0) or len(np.unique(X.index)) > 1:
        X = X.copy(deep=True)
        X.index = np.arange(0, len(X.index))
    return X


def stdp(X):
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
                sum_sq += value ** 2  # todo missing values NaN messes
                # this up!
    mean = sum / num_values
    stdp = np.math.sqrt(sum_sq / num_values - mean ** 2)
    return stdp


def bin_instances_by_class(X, class_labels):
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


def max_instance_dimension_length(X, dimension):
    num_instances = X.shape[0]
    max = -1
    for instance_index in range(0, num_instances):
        instance = X.iloc[instance_index, dimension]
        if len(instance) > max:
            max = len(instance)
    return max


def max_instance_length(X):
    # todo use all dimensions / uneven length dataset
    max_length = len(X.iloc[0, 0])
    # max = -1
    # for dimension in range(0, instances.shape[1]):
    #     length = max_instance_dimension_length(instances, dimension)
    #     if length > max:
    #         max = length
    return max_length
