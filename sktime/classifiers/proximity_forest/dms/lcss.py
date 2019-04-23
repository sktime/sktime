from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
from classifiers.proximity_forest.dms.dtw import Dtw

from datasets import load_gunpoint
from distances.elastic_cython import lcss_distance


class Lcss(Dtw):

    epsilon_key = 'epsilon'
    default_epsilon = 0.01

    def __init__(self, **params):
        self._epsilon = None
        super(Lcss, self).__init__(**params)

    def distance(self, a, b, cut_off):
        return lcss_distance(a, b, **self.get_params())

    def set_params(self, **params):
        super(Lcss, self).set_params(**params)
        super(Lcss, self)._set_param(self.epsilon_key, self.default_epsilon, params)

    def get_params(self):
        return {self.epsilon_key: self._epsilon, **super(Lcss, self).get_params()}

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)
