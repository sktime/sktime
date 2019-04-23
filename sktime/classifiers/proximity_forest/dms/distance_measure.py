from pandas import DataFrame, Series

from classifiers.proximity_forest.parameterised import Parameterised
from datasets import load_gunpoint


class DistanceMeasure(Parameterised):
    def __init__(self, **params):
        if type(self) is DistanceMeasure:
            raise Exception('this is an abstract class')
        super(DistanceMeasure, self).__init__(**params)

    def distance(self, a, b, cut_off):
        raise Exception('this is an abstract class')


if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)
    print(type(x_train.iloc[:,0]))