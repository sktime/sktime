import numpy as np
import matplotlib.pyplot as plt
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

def plot_cost_matrix_color_map(cost_matrix):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()
    min_val, max_val = 0, cost_matrix.shape[0]
    ax.matshow(cost_matrix, cmap='ocean')

    for i in range(max_val):
        for j in range(max_val):
            c = cost_matrix[j, i]
            ax.text(i, j, str(round(c, 2)), va='center', ha='center', size=6)
    plt.show()

def plot_cost_matrix_color_map_path(cost_matrix, path):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()
    min_val, max_val = 0, cost_matrix.shape[0]

    plot_matrix = np.zeros_like(cost_matrix)

    for i in range(max_val):
        for j in range(max_val):
            if (i, j) in path:
                plot_matrix[i, j] = 1.0
            else:
                plot_matrix[i, j] = 0.0

    ax.matshow(plot_matrix, cmap='ocean')

    for i in range(max_val):
        for j in range(max_val):
            c = cost_matrix[i, j]
            ax.text(i, j, str(round(c, 2)), va='center', ha='center', size=6)
    plt.show()


def plot_paths():
    path, distance, cost_matrix = \
        distance_path(x, y, metric=METRIC, return_cost_matrix=True)
    # plot_cost_matrix_color_map(cost_matrix)
    plot_cost_matrix_color_map_path(cost_matrix, path)

    joe = ''
    pass

if __name__ == '__main__':
    plot_paths()


