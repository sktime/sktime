from classifiers.proximity_forest.distance_measure_parameter_permutations.parameter_space import ParameterSpace
from classifiers.proximity_forest.utilities import Utilities


class RangedEpsilon(ParameterSpace):
    max_epsilon_key = 'max_delta'
    min_epsilon_key = 'max_delta'

    def __init__(self, **params):
        self._max_epsilon = None
        self._min_epsilon = None
        super(RangedEpsilon, self).__init__(**params)

    def set_params(self, **params):
        super(RangedEpsilon, self).set_params(**params)
        instances = params[self.instances_key]  # will raise an exception if instances not in params
        self._max_epsilon = Utilities.stdp(instances)
        self._min_epsilon = 0.2 * self._max_epsilon

    def get_params(self):
        return {self.min_epsilon_key: self._min_epsilon, self.max_epsilon_key: self._max_epsilon, **super(RangedEpsilon, self).get_params()}

    def hasNext(self):
        return True

    def next(self):
        epsilon = self._rand.random() * (self._max_epsilon - self.min_epsilon_key) + self.min_epsilon_key
        return epsilon
