from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
from classifiers.proximity_forest.dms.dtw import Dtw

from datasets import load_gunpoint

# from distances.elastic_cython import twe_distance


class Twe(DistanceMeasure):

    nu_key = 'nu'
    default_nu = 0.01
    lambda_key = 'lambda'
    default_lambda = 1 # todo default vals from paper

    def __init__(self, **params):
        self._nu = None
        self._lambda = None
        super(Twe, self).__init__(**params)

    def find_distance(self, time_series_a, time_series_b, cut_off):
        raise Exception('twe not implemented yet')
        # return twe_distance(time_series_a, time_series_b, **self.get_params())

    def set_params(self, **params):
        super(Twe, self).set_params(**params)
        super(Twe, self)._set_param(self.nu_key, self.default_nu, params)
        super(Twe, self)._set_param(self.lambda_key, self.default_lambda, params)

    def get_params(self):
        return {self.nu_key: self._nu, self.lambda_key: self._lambda}

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)
