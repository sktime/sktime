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