import numpy as np


class ArrayHelper(object):
    @classmethod
    def to_array(cls, values=None, type=float, give_shape=False):
        if values is None:
            if give_shape:
                return np.zeros((0, 0))
            return np.asarray([])
        if isinstance(values, int) or isinstance(values, float):
            return np.asarray([values], type)
        return np.asarray(values, type)

    @classmethod
    def make_one_and_zeroes_vector(cls, size, one=1, one_position='begin'):
        # makes vector of the form [one,0,0,0]
        v = np.asarray([0.0] * size, dtype=float)

        if one_position == 'begin':
            one_position = 0
        else:  # 'end'
            one_position = size - 1

        if size > 0:
            v[one_position] = one
        return v
