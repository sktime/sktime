import numpy as np

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

# find the maximum length of an instance from a set of instances
def max_instance_length(instances, dimension = 0):
    num_instances = instances.shape[0]
    max = -1
    for instance_index in range(0, num_instances):
        instance = instances.iloc[instance_index, dimension]
        if len(instance) > max:
            max = len(instance)
    return max
