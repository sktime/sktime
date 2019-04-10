from classifiers.proximity_forest.distance_measure_parameter_permutations.parameter_space import ParameterSpace
from classifiers.proximity_forest.utilities import Utilities


class RangedG(ParameterSpace):

    def __init__(self, **params):
        super(RangedG, self).__init__(**params)

    def set_params(self, **params):
        super(RangedG, self).set_params(**params)

    def get_params(self):
        return {self.min_g_key: self._min_g, self.max_g_key: self._max_g, **super(RangedG, self).get_params()}

    def hasNext(self):
        return True

    def next(self):
        g = self.get_rand().random()
        return g
