from classifiers.proximity_forest.distance_measure_parameter_permutations.parameter_space import ParameterSpace


class RangedDelta(ParameterSpace):
    max_delta_key = 'max_delta'

    def __init__(self, **params):
        self._max_delta = None
        super(RangedDelta, self).__init__(**params)

    def set_params(self, **params):
        super(RangedDelta, self).set_params(**params)
        instances = params[self.instances_key]  # will raise an exception if instances not in params
        instance = instances.iloc[0, 0]
        self._max_delta = instance.shape[0] * 0.25

    def get_params(self):
        return {self.max_delta_key: self._max_delta, **super(RangedDelta, self).get_params()}

    def hasNext(self):
        return True

    def next(self):
        delta = self._rand.random() * self._max_delta  # todo should delta be rounded to integer?
        return delta
