from classifiers.proximity_forest.distance_measure_parameter_permutations.parameter_space import ParameterSpace
from classifiers.proximity_forest.utilities import Utilities


class RangedG(ParameterSpace):
    max_g_key = 'max_g'
    min_g_key = 'max_g'

    def __init__(self, **params):
        self._max_g = None
        self._min_g = None
        super(RangedG, self).__init__(**params)

    def set_params(self, **params):
        super(RangedG, self).set_params(**params)
        instances = params[self.instances_key]  # will raise an exception if instances not in params
        stdp = Utilities.stdp(instances)
        self._max_g = stdp * 0.8
        self._min_g = stdp * 0.2

    def get_params(self):
        return {self.min_g_key: self._min_g, self.max_g_key: self._max_g, **super(RangedG, self).get_params()}

    def hasNext(self):
        return True

    def next(self):
        g = self._rand.random() * (self._max_g - self.min_g_key) + self.min_g_key
        return g
