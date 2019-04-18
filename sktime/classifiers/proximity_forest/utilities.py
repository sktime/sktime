import numpy as np


class Utilities:
    'Utilities for common behaviour'

    @staticmethod
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

    @staticmethod
    def arg_min(array, rand):
        return rand.choice(Utilities.arg_min(array))

    @staticmethod
    def arg_min(array):
        min_indices = [0]
        min = array[0]
        for index in range(1, len(array)):
            value = array[index]
            if value <= min:
                if value < min:
                    min_indices = []
                    min = value
                min_indices.append(index)
        return min_indices

    @staticmethod
    def bin_instances_by_class(instances, class_labels):
        bins = {}
        for class_label in np.unique(class_labels):
            bins[class_label] = []
        num_instances = instances.shape[0]
        for instance_index in range(0, num_instances):
            instance = instances.iloc[instance_index, :]
            class_label = class_labels[instance_index]
            bin = bins[class_label]
            bin.append(instance)
        return bins
