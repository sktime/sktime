from classifiers.proximity_forest.parameterised import Parameterised
from classifiers.proximity_forest.randomised import Randomised


class ParameterSpace(Parameterised, Randomised):

    instances_key = 'instances'

    def __init__(self, **params):
        if type(self) is ParameterSpace:
            raise Exception('this is an abstract class')
        super(ParameterSpace, self).__init__(**params)

    def hasNext(self): # for iterating through param space without replacement
        raise Exception('this is an abstract class')

    def next(self):
        raise Exception('this is an abstract class')