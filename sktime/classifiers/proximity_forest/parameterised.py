import warnings


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

    def _set_param(self, name, default_value, params, prefix='', suffix=''):
        name = prefix + name + suffix
        try:
            value = params[name]
        except:
            warnings.warn('using default value for ' + name)
            value = default_value
        setattr(self, name, value)