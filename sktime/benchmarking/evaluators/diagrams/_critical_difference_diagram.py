# -*- coding: utf-8 -*-
"""Generate critical difference diagrams."""
import math
import warnings
from itertools import combinations
from operator import itemgetter
from typing import Generator, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings(
    "ignore"
)  # Hide warnings that can generate and clutter notebook


def create_critical_difference_diagram(
    df,
    output_path=None,
    title=None,
    alpha: float = 0.05,
    color: str = "0.0",
    space_between_labels: float = 0.15,
    fontsize: float = 15,
    title_fontsize: float = 20,
    figure_width: float = None,
    figure_height: float = None,
) -> Union[plt.Figure, List[plt.Figure]]:
    """Create a critical difference diagram.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 and onwards should be the
         estimators metric scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | metric1  | metric2 |
        | cls1      | data1   | 1.2      | 1.2     |
        | cls2      | data2   | 3.4      | 1.4     |
        | cls1      | data2   | 1.4      | 1.3     |
        | cls2      | data1   | 1.3      | 1.2     |
        ----------------------------------
        Each metric will have its own critical difference diagram produced for it.
    output_path: str, defaults = None
        String that is the path to output the figure. If not specified then figure
        isn't written
    title: str, defaults = None
        Title for figures. If not specified, no title is given then the figure
        is named the same as the column provided. If you want no title simply
        specify title = ''.
    alpha: float, defaults = 0.05
        Alpha value to use to reject Holm hypothesis.
    color: str, defaults = '0.0'
        The matplotlib color of the lines.
    space_between_labels: float, defaults = 0.25
        The space that is between each estimator's line.
    fontsize: float, defaults = 15
        Fontsize for text in graphic
    title_fontsize: float, defaults = 20
        Fontsize for title of graphic
    figure_width: float, defaults = None
        Width of the figure. If not set then will be automatically defined.
    figure_height: float, defaults = None
        Height of figure. If not set then will be automatically defined.

    Returns
    -------
    plt.Figure or List[plt.Figure]
        If more than one metric passed then a list of critical difference diagram
        figures is return else plt.Figure is returned.
    """
    num_metrics = len(df.columns) - 2
    figures = []
    for i in range(2, num_metrics + 2):
        graph_title = title
        if title is None:
            graph_title = df.columns[i]
        curr_df = df.iloc[:, 0:2]
        curr_df["metric"] = df.iloc[:, i]
        avg_ranks = _compute_average_rank(curr_df)
        p_values = _compute_wilcoxon_signed_rank(curr_df, alpha)
        cliques = form_cliques(p_values, avg_ranks.keys())

        figure = _plot_critical_difference_diagram(
            estimators=list(avg_ranks.keys()),
            ranks=list(avg_ranks.values),
            cliques=cliques,
            color=color,
            space_between_labels=space_between_labels,
            title=graph_title,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
            figure_width=figure_width,
            figure_height=figure_height,
        )

        if output_path is not None:
            curr_path = output_path
            if (
                ".jpg" not in output_path
                and ".png" not in output_path
                and ".jpeg" not in output_path
            ):
                # Means they've not given an actual file name
                curr_path = f"{output_path}/{graph_title}.png"
            figure.savefig(curr_path)
        figures.append(figure)

    if len(figures) == 1:
        return figures[0]
    return figures


def _find_edges(graph: List[List[int]]) -> List[Tuple]:
    """Find edges of a graph.

    Parameters
    ----------
    graph: List[List[int]]
        A 2d list containing the graph.

    Returns
    -------
    List[Tuple]
        List of tuples containing edges of graph.
    """
    edges = []
    for i in range(len(graph)):
        curr_node = graph[i]
        for j in range(len(curr_node)):
            curr_edge = curr_node[j]
            if curr_edge == 1:
                edges.append((i, j))
    return edges


def _k_cliques_generator(graph: List[List[int]]) -> Generator:
    """Create a generator for cliques in graph.

    Parameters
    ----------
    graph: List[List[int]]
        A 2d list containing the graph.

    Returns
    -------
    Generator
        Generator that yields each clique.
    """
    edges = _find_edges(graph)
    cliques = [{i, j} for i, j in edges if i != j]
    k = 2

    while cliques:
        yield k, cliques
        cliques_1 = set()
        for u, v in combinations(cliques, 2):
            w = u ^ v
            if len(w) == 2 and tuple(w) in edges:
                cliques_1.add(tuple(u | w))

        cliques = list(map(set, cliques_1))
        k += 1

    return cliques


def _k_cliques(graph: List[List[int]]) -> List[List[int]]:
    """Find cliques in graph.

    Parameters
    ----------
    graph: List[List[int]]
        A 2d list containing the graph.

    Returns
    -------
    List[List[int]]
        List containing the valid cliques for graph.
    """
    cliques = list(_k_cliques_generator(graph))
    valid_cliques = []
    checked_off = []

    for clique in reversed(cliques):
        for curr_clique in clique[1]:
            curr = list(curr_clique)
            found = False
            curr_checked = []
            for val in curr:
                if val in checked_off:
                    found = True
                curr_checked.append(val)

            if found is False:
                checked_off = checked_off + curr_checked
                valid_cliques.append(curr)

    return valid_cliques


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

    return _k_cliques(graph)


def _compute_wilcoxon_signed_rank(
    df: pd.DataFrame,
    alpha=0.05,
) -> List[Tuple]:
    """Compute the wilcoxon signed rank for a dataframe of result.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 and onwards should be the
         estimators metric scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | metric1  |
        | cls1      | data1   | 1.2      |
        | cls2      | data2   | 3.4      |
        | cls1      | data2   | 1.4      |
        | cls2      | data1   | 1.3      |
        ----------------------------------
    alpha: float, defaults = 0.05
        Alpha values to use to reject Holm hypothesis.

    Returns
    -------
    List[Tuple]
        List of tuples of length 4. Where index 0 is the name of the first estimator,
        index 1 is the name of the estimator it was compared to, index 2 is the p value
        and index 3 is a boolean that when true means two classifiers are not critically
        different and false means they are critically different.
    """
    estimators = (df.iloc[:, 0]).unique()

    acc_arr = []
    for estimator in estimators:
        acc_arr.append(
            (df.loc[df[df.columns[0]] == estimator][df.columns[-1]]).to_numpy()
        )

    if len(acc_arr) >= 3:
        friedman_p_value = friedmanchisquare(*acc_arr)[1]

        if friedman_p_value >= alpha:
            raise ValueError(
                "The estimators results provided cannot reject the null" "hypothesis."
            )
    p_values = []

    for i in range(len(estimators)):
        curr_estimator = estimators[i]
        curr_estimator_accuracy = (
            df.loc[df[df.columns[0]] == curr_estimator][df.columns[-1]]
        ).to_numpy()
        for j in range(i + 1, len(estimators)):
            curr_compare_estimator = estimators[j]
            curr_compare_estimator_accuracy = (
                df.loc[df[df.columns[0]] == curr_compare_estimator][df.columns[-1]]
            ).to_numpy()

            p_value = wilcoxon(
                curr_estimator_accuracy,
                curr_compare_estimator_accuracy,
                zero_method="pratt",
            )[1]

            p_values.append((curr_estimator, curr_compare_estimator, p_value, False))

    p_values.sort(key=itemgetter(2))

    for i in range(len(p_values)):
        new_alpha = float(alpha / (len(p_values) - i))
        if not p_values[i][2] <= new_alpha:
            break
        p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)

    return p_values


def _compute_average_rank(df: pd.DataFrame) -> pd.Series:
    """Compute the average ranking for each estimator.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 and onwards should be the
         estimators metric scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | metric1  | metric2 |
        | cls1      | data1   | 1.2      | 1.2     |
        | cls2      | data2   | 3.4      | 1.4     |
        | cls1      | data2   | 1.4      | 1.3     |
        | cls2      | data1   | 1.3      | 1.2     |
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

    sorted_df = df.loc[df[df.columns[0]].isin(estimators)].sort_values(
        [df.columns[0], df.columns[1]]
    )
    rank_data = np.array(sorted_df[df.columns[-1]]).reshape(
        len(estimators), num_datasets
    )

    df_ranks = pd.DataFrame(
        data=rank_data,
        index=np.sort(estimators),
        columns=np.unique(sorted_df[df.columns[1]]),
    )

    ranking_values = (
        df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    )

    return ranking_values


def _plot_critical_difference_diagram(
    estimators: List[str],
    ranks: List[float],
    cliques: List[List[int]] = None,
    color: str = "0.0",
    space_between_labels: float = 0.15,
    title: str = "",
    fontsize: float = 15,
    title_fontsize: float = 20,
    figure_width: float = None,
    figure_height: float = None,
):
    """Plot the critical difference diagram.

    Parameters
    ----------
    estimators: List[str]
        List of string that is the estimators to plot.
    ranks: List[str]
        List of floats that is the ranking for each estimator. The indexes should map
        to the indexes of the estimator's parameter.
    cliques: List[List[int]], defaults = []
        List where each inner List is a list of ints that represents the indexes of the
        estimators that are not critically different from one another.
    color: str, defaults = '0.0'
        The matplotlib color of the lines.
    space_between_labels: float, defaults = 0.25
        Space between each estimator's line.
    title: str, defaults = '
        Title for the critical difference diagram
    fontsize: float, defaults = 15
        Fontsize for text in graphic
    title_fontsize: float, defaults = 20
        Fontsize for title of graphic
    figure_width: float, defaults = None
        Width of the figure. If not set then will be automatically defined.
    figure_height: float, defaults = None
        Height of figure. If not set then will be automatically defined.

    Returns
    -------
    plt.Figure
        Figure containing the critical difference diagram.
    """
    if cliques is None:
        cliques = []
    # Get the scale i.e 1 - x
    min_rank = 1
    max_rank = math.ceil(max(ranks) + (min(ranks) - 1))

    # Get the labels positions
    num_labels_left = math.floor(len(estimators) / 2)
    num_labels_right = math.ceil(len(estimators) / 2)

    label_distance = max_rank / 2
    labels_x_right = [min_rank - label_distance] * num_labels_left
    labels_x_left = [max_rank + label_distance] * num_labels_right
    labels_x_positions = labels_x_left + labels_x_right

    number_line_y = num_labels_right

    labels_y_positions = []

    # This is also dependent on the number of cliques
    num_cliques = 0
    for clique in cliques:
        if len(clique) > 1:
            num_cliques += 1

    cliques_modifier = num_cliques * 0.1
    start = number_line_y - space_between_labels - cliques_modifier
    for _ in range(len(labels_x_left)):
        labels_y_positions.append(start)
        start -= space_between_labels

    start = (
        number_line_y - (space_between_labels * len(labels_x_right)) - cliques_modifier
    )
    for _ in range(len(labels_x_right)):
        labels_y_positions.append(start)
        start += space_between_labels

    x_point_on_line = np.array(ranks)
    y_point_on_line = np.array([number_line_y] * len(ranks))
    fig, ax = plt.subplots()

    min_y = min(labels_y_positions)
    max_y = number_line_y + space_between_labels

    ax.set_ylim([min_y, max_y])

    ax.scatter(labels_x_positions, labels_y_positions, s=0)
    ax.scatter(x_point_on_line, y_point_on_line, c="g", alpha=0.5, s=0)

    # Plot the number line
    start = 1
    end = math.ceil(max(ranks) + (min(ranks) - 1))
    ax.plot(
        np.array([start, end]), np.array([number_line_y, number_line_y]), color=color
    )

    # This will draw the right angle lines to connect labels
    CONNECTION_LINE = "angle,angleA=360,angleB=90,rad=0"
    for i in range(len(estimators)):
        estimator = estimators[i]

        if i > num_labels_left or (
            num_labels_left == num_labels_right and i >= num_labels_left
        ):
            ha = "right"
            estimator_name_ha = "left"
            x_pos_modifier = -20
        else:
            ha = "left"
            estimator_name_ha = "right"
            x_pos_modifier = 20

        ax.annotate(
            "",
            xy=(x_point_on_line[i], y_point_on_line[i]),
            xytext=(labels_x_positions[i], labels_y_positions[i]),
            fontsize=fontsize,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                shrinkA=25,
                shrinkB=0,
                patchA=None,
                patchB=None,
                connectionstyle=CONNECTION_LINE,
            ),
            ha=estimator_name_ha,
        )

        x_pos = x_pos_modifier
        y_pos = -5

        ax.annotate(
            estimator,
            xy=(labels_x_positions[i], labels_y_positions[i]),
            xytext=(x_pos, y_pos),
            textcoords="offset points",
            fontsize=fontsize,
            ha=estimator_name_ha,
        )

        if x_pos > 0:
            rank_x_pos = x_pos + 5
        else:
            rank_x_pos = x_pos - 5

        ax.annotate(
            str(round(ranks[i], 3)),
            xy=(labels_x_positions[i], labels_y_positions[i]),
            xytext=(rank_x_pos, 1),
            textcoords="offset points",
            fontsize=fontsize,
            ha=ha,
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
                int(curr), (curr, number_line_y + 0.06), ha="center", fontsize=fontsize
            )
            ax.plot([curr, curr], [number_line_y, number_line_y + 0.05], color=color)
        else:
            ax.plot([curr, curr], [number_line_y, number_line_y + 0.015], color=color)

    ax.set_title(title, fontsize=title_fontsize)
    plt.gca().invert_xaxis()

    plt.axis("off")
    fig.tight_layout()

    if figure_width is None:
        fig.set_figwidth(number_line_y + 5)
    else:
        fig.set_figwidth(figure_width)

    if figure_height is not None:
        fig.set_figheight(figure_height)

    return fig
