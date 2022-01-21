# from sktime.datasets import load_italy_power_demand as load
#from sktime.datasets import load_gunpoint as load
from sktime.datasets import load_arrow_head as load
# from sktime.datasets import load_basic_motions as load

import numpy as np


from sktime.distances.tests._utils import create_test_distance_numpy
from sktime.distances import distance_factory

x_train, y_train = load("train", True)
x_test, y_test = load("test", True)

first = x_train.iloc[0][0].to_numpy()
second = x_train.iloc[1][0].to_numpy()

cvalues = [0, 0.01, 0.1, 1, 10]

for c_val in cvalues:
    msm = distance_factory(metric="msm", c=c_val)
    dist = msm(first, second)
    print("c: "+str(dist))

x = create_test_distance_numpy(10, 1)
y = create_test_distance_numpy(10, 1, random_state=2)

# it breaks on this as the msm implementation is univariate - do we have a dependent definition of msm?
msm = distance_factory(metric="msm")
dist = msm(x, y)
print(dist)