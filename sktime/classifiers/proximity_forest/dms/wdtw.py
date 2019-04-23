from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
from classifiers.proximity_forest.dms.dtw import Dtw

from datasets import load_gunpoint
from distances.elastic_cython import wdtw_distance


class Wdtw(Dtw):

    g_key = 'g'
    default_g = 0.01

    def __init__(self, **params):
        self._g = None
        super(Wdtw, self).__init__(**params)

    def find_distance(self, a, b, cut_off):
        return wdtw_distance(a, b, **self.get_params())

    def set_params(self, **params):
        super(Wdtw, self).set_params(**params)
        super(Wdtw, self)._set_param(self.g_key, self.default_g, params)

    def get_params(self):
        return {self.g_key: self._g, **super(Wdtw, self).get_params()}

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)
