import numpy as np
from elastic_cython import dtw_distance


if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape((1, 5))
    b = np.array([3.0,4.0,3.0,6.0,1.0]).reshape((1, 5))
    dist = dtw_distance(a, b, w=5)
    print(dist)


