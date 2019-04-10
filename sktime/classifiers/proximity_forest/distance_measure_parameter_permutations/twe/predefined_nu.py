from classifiers.proximity_forest.distance_measure_parameter_permutations.parameter_space import ParameterSpace

import numpy as np


class PredefinedNu(ParameterSpace):
    nu_values_key = 'nu_values'

    def __init__(self, **params):
        self._nu_values = np.array([])
        super(PredefinedNu, self).__init__(**params)

    def set_params(self, **params):
        super(PredefinedNu, self).set_params(**params)
        self._nu_values = np.array(
            [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])

    def get_params(self):
        return {self.nu_values_key: self._nu_values, **super(PredefinedNu, self).get_params()}

    def hasNext(self):
        return True

    def next(self):
        rand = self.get_rand()
        return rand.choice(self._nu_values)
