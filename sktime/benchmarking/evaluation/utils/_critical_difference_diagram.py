from typing import List, Tuple
import math
import Orange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import networkx
from operator import itemgetter


from scipy.stats import friedmanchisquare, wilcoxon


def create_critical_difference_diagram(df, output_path, title):
    """Creates a critical difference diagram.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 should be the accuracy the
        estimator scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | accuracy |
        | cls1      | data1   | 1.2      |
        | cls2      | data2   | 3.4      |
        | cls1      | data2   | 1.4      |
        | cls2      | data1   | 1.3      |
        ----------------------------------
    """
    avg_ranks = compute_average_rank(df)
    p_values = compute_wilcoxon_signed_rank(df)

    cliques = form_cliques(p_values, avg_ranks.keys())

    estimators = list(avg_ranks.keys())
    ranks = list(avg_ranks.values)
    cliques = cliques
    joe = ''

    figure_1 = _plot_critical_difference_diagram(
        estimators=list(avg_ranks.keys()),
        ranks=list(avg_ranks.values),
        cliques=cliques
    )

    # figure_2 = _plot_critical_difference_diagram(estimators=)



    joe = ''


def form_cliques(p_values, estimators) -> List[List[int]]:
    """For clique for critical difference.

    This method is used to find grouping that are not critically different so that they
    can be connected in the final graph.

    Parameters
    ----------
    p_values: List[Tuple]
        List of tuples of length 4. Where index 0 is the name of the first estimator,
        index 1 is the name of the estimator it was compared to, index 2 is the p value
        and index 3 is a boolean that when true means two classifiers are not critically
        different and false means they are critically different.
    estimators: Pd.Index
        Index of keys that are the estimators.

    Returns
    -------
    List[List[int]]
        List where each list contains the index of the estimators that are not
        critically different.
    """
    m = len(estimators)
    graph = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if not p[3]:
            i = np.where(estimators == p[0])[0]
            j = np.where(estimators == p[1])[0]
            min_i = min(i, j)
            max_j = max(i, j)
            graph[min_i, max_j] = 1

    return list(networkx.find_cliques(networkx.Graph(graph))) # TODO: rewrite this function nativly in sktime.


def compute_wilcoxon_signed_rank(df: pd.DataFrame, alpha=0.05) -> List[Tuple]:
    """Compute the wilcoxon signed rank for a dataframe of result.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 should be the accuracy the
        estimator scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | accuracy |
        | cls1      | data1   | 1.2      |
        | cls2      | data2   | 3.4      |
        | cls1      | data2   | 1.4      |
        | cls2      | data1   | 1.3      |
        ----------------------------------
    alpha: float
        Alpha value to use to reject Holm hypotheiss.

    Returns
    -------
    List[Tuple]
        List of tuples of length 4. Where index 0 is the name of the first estimator,
        index 1 is the name of the estimator it was compared to, index 2 is the p value
        and index 3 is a boolean that when true means two classifiers are not critically
        different and false means they are critically different.
    """
    datasets = (df.iloc[:, 1]).unique()
    num_datasets = len(datasets)
    estimators = (df.iloc[:, 0]).unique()

    acc_arr = []
    for estimator in estimators:
        acc_arr.append((df.loc[df[df.columns[0]] == estimator][df.columns[-1]])
                       .to_numpy())

    friedman_p_value = friedmanchisquare(*acc_arr)[1]

    if friedman_p_value >= alpha:
        raise ValueError("The estimators results provided cannot reject the null"
                         "hypothesis.")
    p_values = []

    for i in range(len(estimators)):
        curr_estimator = estimators[i]
        curr_estimator_accuracy = \
            (df.loc[df[df.columns[0]] == curr_estimator][df.columns[-1]]).to_numpy()
        for j in range(i + 1, len(estimators)):
            curr_compare_estimator = estimators[j]
            curr_compare_estimator_accuracy = \
                (df.loc[df[df.columns[0]] == curr_compare_estimator][df.columns[-1]]) \
                    .to_numpy()

            p_value = wilcoxon(
                curr_estimator_accuracy,
                curr_compare_estimator_accuracy,
                zero_method='pratt'
            )[1]

            p_values.append((curr_estimator, curr_compare_estimator, p_value, False))

    p_values.sort(key=operator.itemgetter(2))

    for i in range(len(p_values)):
        new_alpha = float(alpha / (len(p_values) - i))
        if not p_values[i][2] <= new_alpha:
            break
        p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)

    return p_values


def compute_average_rank(df: pd.DataFrame) -> pd.Series:
    """Compute the average ranking for each estimator.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 should be the accuracy the
        estimator scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | accuracy |
        | cls1      | data1   | 1.2      |
        | cls2      | data2   | 3.4      |
        | cls1      | data2   | 1.4      |
        | cls2      | data1   | 1.3      |
        ----------------------------------
    Returns
    -------
    Pd.Series
        Series where each element is a series where the classifier is index 0 and the
        rank is index 1.
    """
    datasets = (df.iloc[:, 1]).unique()
    num_datasets = len(datasets)
    estimators = (df.iloc[:, 0]).unique()

    sorted_df = df.loc[df[df.columns[0]].isin(estimators)]. \
        sort_values([df.columns[0], df.columns[1]])
    rank_data = np.array(sorted_df[df.columns[-1]]).reshape(
        len(estimators), num_datasets
    )

    df_ranks = pd.DataFrame(
        data=rank_data,
        index=np.sort(estimators),
        columns=np.unique(sorted_df[df.columns[1]])
    )

    ranking_values = \
        df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)

    return ranking_values


def _plot_critical_difference_diagram(
        estimators: List[str],
        ranks: List[float],
        cliques: List[List[int]] = [],
        color: str = '0.0',
        space_between_labels: float = 0.15,
        title: str = '',
        fontsize: float = 15,
        title_fontsize: float = 20
):
    """Plot the critical difference diagram.

    Parameters
    ----------
    estimators: List[str]
        List of string that is the estimators to plot.
    ranks: List[str]
        List of floats that is the ranking for each estimator. The indexes should map
        to the indexes of the estimators parameter.
    cliques: List[List[int]], defaults = []
        List where each inner List is a list of ints that represents the indexes of the
        estimators that are not critically different from one another.
    color: str, defaults = '0.0'
        The matplotlib color of the lines.
    title: str, defaults = '
        Title for the critical difference diagram
    fontsize: float, defaults = 15
        Fontsize for text in graphic
    title_fontsize: float, defaults = 20
        Fontsize for title of graphic

    Returns
    -------
    plt.Figure
        Figure containing the critical difference diagram.
    """
    # Get the scale i.e 1 - x
    min_rank = 1
    max_rank = math.ceil(max(ranks) + (min(ranks) - 1))

    # Get the labels positions
    num_labels_left = math.floor(len(estimators) / 2)
    num_labels_right = math.ceil(len(estimators) / 2)

    labels_x_right = [min_rank - 2] * num_labels_left
    labels_x_left = [max_rank + 2] * num_labels_right
    labels_x_positions = labels_x_left + labels_x_right

    number_line_y = num_labels_right

    labels_y_positions = []

    # This is also dependent on the number of cliques
    cliques_modifier = len(cliques) * 0.1
    start = number_line_y - space_between_labels - cliques_modifier
    for _ in range(len(labels_x_left)):
        labels_y_positions.append(start)
        start -= space_between_labels

    start = number_line_y - (space_between_labels * len(labels_x_right)) - cliques_modifier
    for _ in range(len(labels_x_right)):
        labels_y_positions.append(start)
        start += space_between_labels

    x_point_on_line = np.array(ranks)
    y_point_on_line = np.array([number_line_y] * len(ranks))
    fig,ax = plt.subplots()

    min_y = min(labels_y_positions)
    max_y = number_line_y + space_between_labels

    ax.set_ylim([min_y, max_y])


    ax.scatter(labels_x_positions, labels_y_positions, s=0)
    ax.scatter(x_point_on_line, y_point_on_line, c="g", alpha=0.5, s=0)

    # Plot the number line
    start = 1
    end = math.ceil(max(ranks) + (min(ranks) - 1))
    ax.plot(
        np.array([start, end]), np.array([number_line_y, number_line_y]),
        color=color
    )


    # This will draw the right angle lines to connect labels
    CONNECTION_LINE = "angle,angleA=360,angleB=90,rad=0"
    for i in range(len(estimators)):
        estimator = estimators[i]
        ax.annotate(
            estimator,
            xy=(x_point_on_line[i], y_point_on_line[i]),
            xytext=(labels_x_positions[i], labels_y_positions[i]),
            fontsize=fontsize,
            arrowprops=dict(arrowstyle="-", color=color,
                            shrinkA=13, shrinkB=1,
                            patchA=None, patchB=None,
                            connectionstyle=CONNECTION_LINE,
                            ),
        )

        if i > num_labels_left:
            ha = 'right'
            x_pos_modifier = 0.06
        else:
            ha = 'left'
            x_pos_modifier = -0.4

        x_pos = labels_x_positions[i] + x_pos_modifier
        y_pos = labels_y_positions[i] + 0.02

        ax.annotate(
            str(round(ranks[i], 3)),
            xy=(x_pos, y_pos),
            xytext=(x_pos, y_pos),
            fontsize=fontsize,
            ha=ha
        )

    # Clique positions
    cliques_y = number_line_y - 0.1
    for clique_x in cliques:
        x = np.array(itemgetter(*clique_x)(ranks))
        y = np.array(([cliques_y] * len(clique_x)))
        ax.plot(x, y, color=color, linewidth=5.0)
        cliques_y -= 0.1


    for i in range(1, end * 2):
        curr = (i * 0.5) + 0.5
        if curr % 1 == 0:
            ax.annotate(
                int(curr), (curr, number_line_y + 0.06), ha='center', fontsize=fontsize
            )
            ax.plot([curr, curr], [number_line_y, number_line_y + 0.05], color=color)
        else:
            ax.plot([curr, curr], [number_line_y, number_line_y + 0.015], color=color)

    ax.set_title(title, fontsize=title_fontsize)
    plt.gca().invert_xaxis()

    plt.axis('off')
    fig.tight_layout()

    return fig


if __name__ == '__main__':
    df = pd.read_csv('./test.csv', index_col=False)
    create_critical_difference_diagram(df, output_path='./', title='Accuracy')
