from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
from classifiers.proximity_forest.dms.dtw import Dtw

from datasets import load_gunpoint
from distances.elastic_cython import ddtw_distance


class Ddtw(Dtw):

    def __init__(self, **params):
        super(Ddtw, self).__init__(**params)

    def find_distance(self, a, b, cut_off):
        return ddtw_distance(a, b, **self.get_params())

    def set_params(self, **params):
        super(Ddtw, self).set_params(**params)

    def get_params(self):
        return super(Ddtw, self).get_params()

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)