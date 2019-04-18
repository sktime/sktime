import numpy as np

from classifiers.proximity_forest.distance_measure_parameter_permutations.dtw_parameter_space import DtwParameterSpace
from classifiers.proximity_forest.dms.lcss import Lcss
from classifiers.proximity_forest.utilities import Utilities
from datasets import load_gunpoint


class LcssParameterSpace(DtwParameterSpace):
    'Lcss parameter space'

    max_epsilon_key = 'max_epsilon'
    min_epsilon_key = 'min_epsilon'

    def __init__(self, **params):
        self._max_epsilon = -1
        self._min_epsilon = -1
        super(LcssParameterSpace, self).__init__(**params)

    def set_params(self, **params):
        super(LcssParameterSpace, self).set_params(**params)
        instances = params[self.instances_key]  # will raise an exception if instances not in params
        self._max_epsilon = Utilities.stdp(instances)
        self._min_epsilon = 0.2 * self._max_epsilon

    def get_params(self):
        return {**super(LcssParameterSpace, self).get_params(), self.max_epsilon_key: self._max_epsilon,
                self.min_epsilon_key: self._min_epsilon}

    def get_random_parameter_permutation(self):
        epsilon = self._rand.random() * (self._max_epsilon - self._min_epsilon) + self._min_epsilon
        return {Lcss.epsilon_key: epsilon, **super(LcssParameterSpace, self).get_random_parameter_permutation()}


if __name__ == "__main__":
    instances, class_labels = load_gunpoint(return_X_y=True)
    ps = LcssParameterSpace(**{LcssParameterSpace.instances_key: instances})
    print(ps.get_params())

