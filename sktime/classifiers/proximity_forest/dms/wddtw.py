from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
from classifiers.proximity_forest.dms.dtw import Dtw
from classifiers.proximity_forest.dms.wdtw import Wdtw

from datasets import load_gunpoint
from distances.elastic_cython import wddtw_distance


class Wddtw(Wdtw):

    def __init__(self, **params):
        super(Wddtw, self).__init__(**params)

    def find_distance(self, a, b, cut_off):
        return wddtw_distance(a, b, **self.get_params())

    def set_params(self, **params):
        super(Wddtw, self).set_params(**params)

    def get_params(self):
        return super(Wddtw, self).get_params()

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)