class Parameterised:
    def __init__(self, **params):
        if type(self) is Parameterised:
            raise Exception('this is an abstract class')
        self.set_params(**params)
        super(Parameterised, self).__init__()

    def get_params(self):
        raise Exception('this is an abstract class')

    def set_params(self, **params):
        raise Exception('this is an abstract class')