import warnings
import numpy as np

from classifiers.proximity_forest.parameterised import Parameterised
from classifiers.proximity_forest.randomised import Randomised


class DistanceMeasureParameterSpace(Parameterised, Randomised):

    instances_key = 'instances'

    def __init__(self, **params):
        if type(self) is DistanceMeasureParameterSpace:
            raise Exception('this is an abstract class')
        super(DistanceMeasureParameterSpace, self).__init__(**params)

    def get_random_parameter_permutation(self):
        raise Exception('this is an abstract method')


