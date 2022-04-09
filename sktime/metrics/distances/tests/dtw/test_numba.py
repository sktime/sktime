import numpy as np
from numba import int32, float32    # import the types
from numba.experimental import jitclass

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jitclass(spec)
class Base(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

spec2 = [
    ('bounding', Base)
]

@jitclass(spec2)
class Joe(object):
    def __init__(self, baseObj):
        self.bounding = baseObj

def test_run():
    test = Base(69)
    test2 = Joe(test)