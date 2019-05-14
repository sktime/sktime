import numpy as np
from pandas import DataFrame

def predict_from_distribution(distributions, rand, label_encoder):
    predictions = np.empty((distributions.shape[0]), dtype = int)
    for instance_index in range(0, predictions.shape[0]):
        distribution = distributions[instance_index]
        prediction = arg_max(distribution, rand)
        predictions[instance_index] = prediction
    predictions = label_encoder.inverse_transform(predictions)
    return predictions


def check_data(instances, class_labels = None):
    if not isinstance(instances, DataFrame):
        raise ValueError("instances not in panda dataframe")
    if class_labels is not None:
        # todo these checks could probs be / is defined elsewhere
        if len(class_labels) != instances.shape[0]:
            raise ValueError("instances not same length as class_labels")

# find the standard deviation of the dataset
def stdp(instances):
    sum = 0
    sum_sq = 0
    num_instances = instances.shape[0]
    num_dimensions = instances.shape[1]
    num_values = num_instances * num_dimensions
    for instance_index in range(0, num_instances):
        for dimension_index in range(0, num_dimensions):
            instance = instances.iloc[instance_index, dimension_index]
            for value in instance:
                sum += value
                sum_sq += (value ** 2)  # todo missing values NaN messes this up!
    mean = sum / num_values
    stdp = np.math.sqrt(sum_sq / num_values - mean ** 2)
    return stdp

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
def mins(array):
    indices = arg_mins(array)
    return pick_from_indices(array, indices)

# find max values in array
def maxs(array):
    indices = arg_maxs(array)
    return pick_from_indices(array, indices)

# find min value in array, randomly breaking ties
def min(array, rand):
    index = arg_min(array, rand)
    return array[index]

# find max value in array, randomly breaking ties
def max(array, rand):
    index = arg_max(array, rand)
    return array[index]

# find best value in array, randomly breaking ties
def best(array, comparator, rand):
    return rand.choice(bests(array, comparator))

# find index of best value in array, randomly breaking ties
def arg_best(array, comparator, rand):
    return rand.choice(arg_bests(array, comparator))

# find index of min value in array, randomly breaking ties
def arg_min(array, rand):
    return rand.choice(arg_mins(array))

# find indices of best value in array, randomly breaking ties
def arg_mins(array):
    return arg_bests(array, less_than)

# find index of max value in array, randomly breaking ties
def arg_max(array, rand):
    return rand.choice(arg_maxs(array))

# find indices of max value in array, randomly breaking ties
def arg_maxs(array):
    return arg_bests(array, more_than)

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

# convert given instances and class labels into dict of class label mapped to instances
def bin_instances_by_class(instances, class_labels):
    bins = {}
    for class_label in np.unique(class_labels):
        bins[class_label] = []
    num_instances = instances.shape[0]
    for instance_index in range(0, num_instances):
        instance = instances.iloc[instance_index, :]
        class_label = class_labels[instance_index]
        instances_bin = bins[class_label]
        instances_bin.append(instance)
    return bins

# find the maximum length of an instance from a set of instances
def max_instance_length(instances):
    num_instances = instances.shape[0]
    max = -1
    for instance_index in range(0, num_instances):
        for dim_index in range(0, instances.shape[1]):
            instance = instances.iloc[instance_index, dim_index]
            if len(instance) > max:
                max = len(instance)
    return max
