import numpy as np

# set dataframe indices to a negative range (-1 to below -1)
def negative_dataframe_indices(X):
    if np.any(X.index >= 0) or len(np.unique(X.index)) > 1:
        X = X.copy(deep = True)
        X.index = np.arange(-1, -len(X.index) - 1, step = -1)
    return X

# set dataframe indices to a positive range (0 to above 0)
def positive_dataframe_indices(X):
    if np.any(X.index < 0) or len(np.unique(X.index)) > 1:
        X = X.copy(deep = True)
        X.index = np.arange(0, len(X.index))
    return X

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

# find the maximum length of an instance from a set of instances for a given dimension
def max_instance_dimension_length(instances, dimension):
    num_instances = instances.shape[0]
    max = -1
    for instance_index in range(0, num_instances):
        instance = instances.iloc[instance_index, dimension]
        if len(instance) > max:
            max = len(instance)
    return max

# find the maximum length of an instance from a set of instances for all dimensions
def max_instance_length(instances):
    # todo use all dimensions / uneven length dataset
    max = len(instances.iloc[0, 0])
    # max = -1
    # for dimension in range(0, instances.shape[1]):
    #     length = max_instance_dimension_length(instances, dimension)
    #     if length > max:
    #         max = length
    return max
