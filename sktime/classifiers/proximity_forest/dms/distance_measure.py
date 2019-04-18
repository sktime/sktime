from classifiers.proximity_forest.parameterised import Parameterised


class DistanceMeasure(Parameterised):
    def __init__(self, **params):
        if type(self) is DistanceMeasure:
            raise Exception('this is an abstract class')
        super(DistanceMeasure, self).__init__(**params)

    def find_distance(self, time_series_a, time_series_b, cut_off):
        raise Exception('this is an abstract class')