from classifiers.proximity_forest.randomised import Randomised


class OnePerClassSelector(Randomised):
    def __init__(self, **params):
        super(OnePerClassSelector, self).__init__(**params)

    def get_params(self):
        raise Exception('not implemented')

    def set_params(self, **params):
        super(OnePerClassSelector, self).set_params(**params)

    def select(self, instances, class_labels): # todo generify
        rand = self.get_rand()
        exemplar_instances = []
        exemplar_class_labels = []
        # todo
        return exemplar_instances, exemplar_class_labels