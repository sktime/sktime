from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
import numpy as np

from distance_measures.elastic import dtw_distance


class Dtw(DistanceMeasure):

    delta_key = 'delta'
    default_delta = 1

    def __init__(self, **params):
        self._delta = -1
        super(Dtw, self).__init__(**params)

    def find_distance(self, time_series_a, time_series_b, cut_off):
        return dtw_distance(time_series_a, time_series_b, **self.get_params())

    def set_params(self, **params):
        self._delta = params.get(self.delta_key, self.default_delta)  # todo warn?

    def get_params(self):
        return {self.delta_key: self._delta}

if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    b = np.array([4, 2, 5, 43, 9], dtype=float)
    dm = Dtw(delta=0.5)
    distance = dm.find_distance(a, b, -1)
    print(distance)