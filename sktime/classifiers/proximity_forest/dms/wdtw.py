from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
from classifiers.proximity_forest.dms.dtw import Dtw

from datasets import load_gunpoint
from distance_measures.elastic import lcss_distance, Wdtw_distance


class Wdtw(Dtw):

    g_key = 'epsilon'
    default_g = 0.01

    def __init__(self, **params):
        self._g = -1
        super(Wdtw, self).__init__(**params)

    def find_distance(self, time_series_a, time_series_b, cut_off):
        return Wdtw_distance(time_series_a, time_series_b, **self.get_params())

    def set_params(self, **params):
        super(Wdtw, self).set_params(**params)
        self._g = params.get(self.g_key, self.default_g) # todo warn?

    def get_params(self):
        return {self.g_key: self._g, **super(Wdtw, self).get_params()}

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)
