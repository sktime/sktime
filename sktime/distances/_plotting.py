import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colorplt


from sktime.distances._distance import distance_path, pairwise_distance

gray_cmap = colorplt.LinearSegmentedColormap.from_list("", ["#c9cacb", "white"])

def _path_mask(cost_matrix, path, ax, theme=gray_cmap):
    plot_matrix = np.zeros_like(cost_matrix)
    max_size = max(cost_matrix.shape)
    for i in range(max_size):
        for j in range(max_size):
            if (i, j) in path:
                plot_matrix[i, j] = 1.0
            elif cost_matrix[i, j] == np.inf:
                plot_matrix[i, j] = 0.0
            else:
                plot_matrix[i, j] = 0.5

    for i in range(max_size):
        for j in range(max_size):
            c = cost_matrix[j, i]
            ax.text(i, j, str(round(c, 3)), va='center', ha='center', size=6)
            ax.text(i, j, str(round(c, 3)), va='center', ha='center', size=6)


    ax.matshow(plot_matrix, cmap=theme)


def _pairwise_path(x, y, metric):
    pw_matrix = pairwise_distance(x, y, metric=metric)
    path = []
    for i in range(pw_matrix.shape[0]):
        for j in range(pw_matrix.shape[1]):
            if i == j:
                path.append((i, j))
    return path, pw_matrix.trace(), pw_matrix


def plot_path(
        x: np.ndarray,
        y: np.ndarray,
        metric: str,
        dist_kwargs: dict = None,
        title: str = ''
):
    if dist_kwargs is None:
        dist_kwargs = {}
    try:
        path, dist, cost_matrix = \
            distance_path(x, y, metric=metric, return_cost_matrix=True, **dist_kwargs)
    except NotImplementedError:
        path, dist, cost_matrix = _pairwise_path(x, y, metric)


    plt.figure(1, figsize=(8, 8))
    x_size = x.shape[0]

    # definitions for the axes
    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02

    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    _path_mask(cost_matrix, path, ax_gram)
    ax_gram.axis("off")
    ax_gram.autoscale(False)
    # ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
    #              linewidth=3.)

    ax_s_x.plot(np.arange(x_size), y, "b-", linewidth=3., color='#818587')
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, x_size - 1))

    ax_s_y.plot(- x, np.arange(x_size), "b-", linewidth=3., color='#818587')
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, x_size - 1))

    ax_s_x.set_title(title, size=10)

    plt.show()

def plot_alignment(
        x,
        y,
        metric,
        dist_kwargs: dict = None,
        title: str = ''
):
    if dist_kwargs is None:
        dist_kwargs = {}
    try:
        path, dist, cost_matrix = \
            distance_path(x, y, metric=metric, return_cost_matrix=True, **dist_kwargs)
    except NotImplementedError:
        path, dist, cost_matrix = _pairwise_path(x, y, metric)

    plt.figure(1, figsize=(8, 8))

    plt.plot(x, "b-", color='black')
    plt.plot(y, "g-", color='black')

    for positions in path:
        try:
            plt.plot([positions[0], positions[1]],
                     [x[positions[0]],
                      y[positions[1]]], '--', color='#818587')
        except:
            continue
    plt.legend()
    plt.title(title)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
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

    plot_path(x, y, 'euclidean')
    plot_path(x, y, 'dtw')
    plot_path(x, y, 'dtw', {'window': 0.2})

    plot_alignment(x, y, 'dtw')
    plot_alignment(x, y, 'dtw', {'window': 0.2})
