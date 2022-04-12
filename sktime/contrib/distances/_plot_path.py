import numpy as np
from sktime.distances import distance_path

x = np.array([
    -0.7553383207,
    0.4460987596,
    1.197682907,
    0.1714334808,
    - 0.5639929213,
    - 0.6891222874,
    - 1.793828873,
    0.06570866314,
    0.2877381702,
    1.633620422
])

y = np.array([
    0.01765193577,
    1.536784164,
    - 0.1413292622,
    - 0.7609346135,
    0.1767363331,
    - 2.192007072,
    0.1933165696,
    0.4648166839,
    0.9444888843,
    - 0.239523623
])

METRIC = 'dtw'

def plot_paths():
    path, distance = distance_path(x, y, metric=METRIC)
    joe = ''
    pass

if __name__ == '__main__':
    plot_paths()


