import numpy as np

class Utilities:
    'Utilities for common behaviour'

    @staticmethod
    def stdp(instances):
        sum = 0
        sumSq = 0
        numInstances = instances.shape[0]
        numDimensions = instances.shape[1]
        numValues = numInstances * numDimensions
        for instanceIndex in range(0, numInstances):
            for dimensionIndex in range(0, numDimensions):
                instance = instances.iloc[instanceIndex, dimensionIndex]
                for value in instance:
                    sum += value
                    sumSq += (value ** 2) # todo missing values NaN messes this up!
        mean = sum / numValues
        stdp = np.math.sqrt(sumSq / numValues - mean ** 2)
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