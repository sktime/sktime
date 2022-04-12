import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from sktime.distances import distance_path
from sktime.datatypes import convert_to

x = np.array([
    -0.7553383207,
    0.4460987596,
    1.197682907,
    0.1714334808,
    0.5639929213,
    0.6891222874,
    1.793828873,
    0.06570866314,
    0.2877381702,
    1.633620422
])

y = np.array([
    0.01765193577,
    1.536784164,
    - 0.1413292622,
    - 0.7609346135,
    - 0.1767363331,
    - 2.192007072,
    - 0.1933165696,
    - 0.4648166839,
    - 0.9444888843,
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

position = {
    'x_ts': (0, 0),
    'y_ts': (0, 1),
    'ed_path': (1, 0),
    'dtw_path': (1, 1)
}

def plot_cost_matrix_color_map_path(x, y):
    path, distance, cost_matrix = \
        distance_path(x, y, metric=METRIC, return_cost_matrix=True, window=0.1)
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # fig, ax = plt.subplots(2, 2)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout()
    min_val, max_val = 0, cost_matrix.shape[0]

    plot_matrix = np.zeros_like(cost_matrix)

    ed_plot_matrix = np.zeros_like(cost_matrix)


    for i in range(max_val):
        for j in range(max_val):
            if (i, j) in path:
                plot_matrix[i, j] = 1.0
            elif cost_matrix[i, j] == np.inf:
                plot_matrix[i, j] = 0.5
            else:
                plot_matrix[i, j] = 0.0

            if i == j:
                ed_plot_matrix[i, j] = 1.0
            elif cost_matrix[i, j] == np.inf:
                ed_plot_matrix[i, j] = 0.5
            else:
                ed_plot_matrix[i, j] = 0.0

    ax[position['dtw_path']].matshow(plot_matrix, cmap='ocean')
    ax[position['ed_path']].matshow(ed_plot_matrix, cmap='ocean')

    ax[position['dtw_path']].set_xlabel('x time series')
    ax[position['ed_path']].set_xlabel('x time series')

    ax[position['dtw_path']].xaxis.set_label_position('top')
    ax[position['ed_path']].xaxis.set_label_position('top')

    ax[position['dtw_path']].set_ylabel('y time series')
    ax[position['ed_path']].set_ylabel('y time series')

    ax[position['dtw_path']].set_title('Dtw warping path')
    ax[position['ed_path']].set_title('Ed warping path')


    for i in range(max_val):
        for j in range(max_val):
            c = cost_matrix[j, i]
            ax[position['dtw_path']].text(i, j, str(round(c, 2)), va='center', ha='center', size=6)
            ax[position['ed_path']].text(i, j, str(round(c, 2)), va='center', ha='center', size=6)


    ax[position['y_ts']].plot(y)
    ax[position['x_ts']].plot(x)

    ax[position['y_ts']].title.set_text('y time series')
    ax[position['x_ts']].title.set_text('x time series')

    # ax[0, 0].axis('off')
    # ax[1, 0].axis('off')
    # ax[0, 1].axis('off')
    #
    # test = y.reshape(y.shape[0], 1)
    #
    # ax[1, 0].plot(test)
    # ax[1,0].flip_xy()

def plot_time_series(x, y):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x)
    ax[1].plot(y)

    ax[0].title.set_text('x time series')
    ax[1].title.set_text('y time series')



def plot_paths():
    # plot_cost_matrix_color_map(cost_matrix)
    plot_cost_matrix_color_map_path(x, y)
    # plot_time_series(x, y)

    # plot_time_series(x)
    # plot_time_series(y)
    plt.show()

    joe = ''
    pass

if __name__ == '__main__':
    plot_paths()


