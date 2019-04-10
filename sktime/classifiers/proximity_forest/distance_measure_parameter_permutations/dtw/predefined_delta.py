from classifiers.proximity_forest.distance_measure_parameter_permutations.parameter_space import ParameterSpace

import numpy as np

class PredefinedDelta(ParameterSpace):

    delta_values_key = 'delta_values'

    def __init__(self, **params):
        self._delta_values = np.array([])
        super(PredefinedDelta, self).__init__(**params)

    def set_params(self, **params):
        super(PredefinedDelta, self).set_params(**params)
        self._delta_values = np.array(params[self.delta_values_key])
        instances = params[self.instances_key]  # will raise an exception if instances not in params
        instance = instances.iloc[0, 0]
        self._delta_values *= instance.shape[0]  # todo should delta be rounded to integer?

    def get_params(self):
        return {self.delta_values_key: self._delta_values, **super(PredefinedDelta, self).get_params()}

    def hasNext(self):
        return True

    def next(self):
        rand = self.get_rand()
        return rand.choice(self._delta_values)
