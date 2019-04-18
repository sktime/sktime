from classifiers.proximity_forest.parameterised import Parameterised


class SplitScore(Parameterised):
    def __init__(self, **params):
        super(SplitScore, self).__init__(**params)

    def score(self):
        raise Exception('this is an abstract method')