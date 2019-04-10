from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
from classifiers.proximity_forest.dms.dtw import Dtw

from datasets import load_gunpoint
from distance_measures.elastic import lcss_distance


class Lcss(Dtw):

    epsilon_key = 'epsilon'
    default_epsilon = 0.01

    def __init__(self, **params):
        self._epsilon = -1
        super(Lcss, self).__init__(**params)

    def find_distance(self, time_series_a, time_series_b, cut_off):
        return lcss_distance(time_series_a, time_series_b, **self.get_params())

    def set_params(self, **params):
        super(Lcss, self).set_params(**params)
        self._epsilon = params.get(self.epsilon_key, self.default_epsilon) # todo warn?

    def get_params(self):
        return {self.epsilon_key: self._epsilon, **super(Lcss, self).get_params()}

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)
