"""Standalone module to compute and plot critical difference diagrams."""

__author__ = ["SveaMeyer13"]

import math

import numpy as np
from scipy.stats import distributions, find_repeats, rankdata

from sktime.utils.dependencies import _check_soft_dependencies

_check_soft_dependencies("matplotlib", severity="warning")


def _check_friedman(n_strategies, n_datasets, ranked_data, alpha):
    """Check whether Friedman test is significant.

    Larger parts of code copied from scipy.

    Arguments
    ---------
    n_strategies : int
      number of strategies to evaluate
    n_datasets : int
      number of datasets classified per strategy
    ranked_data : np.array (shape: n_strategies * n_datasets)
      rank of strategy on dataset

    Returns
    -------
    is_significant : bool
      Indicates whether strategies differ significantly in terms of performance
      (according to Friedman test).
    """
    if n_strategies < 3:
        raise ValueError(
            "At least 3 sets of measurements must be given for Friedmann test, "
            f"got {n_strategies}."
        )

    # calculate c to correct chisq for ties:
    ties = 0
    for i in range(n_datasets):
        replist, repnum = find_repeats(ranked_data[i])
        for t in repnum:
            ties += t * (t * t - 1)
    c = 1 - ties / (n_strategies * (n_strategies * n_strategies - 1) * n_datasets)

    ssbn = np.sum(ranked_data.sum(axis=0) ** 2)
    chisq = (
        12.0 / (n_strategies * n_datasets * (n_strategies + 1)) * ssbn
        - 3 * n_datasets * (n_strategies + 1)
    ) / c
    p = distributions.chi2.sf(chisq, n_strategies - 1)
    if p < alpha:
        is_significant = True
    else:
        is_significant = False
    return is_significant


def plot_critical_difference(
    scores,
    labels,
    cliques=None,
    is_errors=True,
    alpha=0.05,
    width=10,
    textspace=2.5,
    reverse=True,
):
    """Draw critical difference diagram.

    Step 1 & 2: Calculate average ranks from data
    Step 3: Use Friedman test to check whether
    the strategy significantly affects the classification performance
    Step 4: Compute critical differences using Nemenyi post-hoc test.
    (How much should the average rank of two strategies differ to be
     statistically significant)
    Step 5: Compute statistically similar cliques of strategies
    Step 6: Draw the diagram

    See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Parts of the code are copied and adapted from here:
    https://github.com/hfawaz/cd-diagram

    Arguments
    ---------
        scores : np.array
            scores (either accuracies or errors) of dataset x strategy
            (best strategy is in most left column)
        labels : list of str
            list with names of the strategies
        cliques : lists of bit vectors,
            e.g. [[0,1,1,1,0,0] [0,0,0,0,1,1]]
            statistically similar cliques of strategies
            optional (default: None, in this case cliques will be computed)
        is_errors : bool
            indicates whether scores are passed as errors (default) or accuracies
        alpha : float (currently supported: 0.1, 0.05 or 0.01)
            Alpha level for statistical tests (default: 0.05)
        width : int
           width in inches (default: 10)
        textspace : int
           space on figure sides (in inches) for the method names (default: 2.5)
        reverse : bool
           if set to 'True', the lowest rank is on the right (default: 'True')
    """
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    # Helper Functions
    def _nth(lst, n):
        """Return only nth element in a list."""
        n = _lloc(lst, n)
        return [a[n] for a in lst]

    def _lloc(lst, n):
        """List location in list of list structure.

        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(lst[0]) + n
        else:
            return n

    def _rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    def _line(lst, color="k", **kwargs):
        """Input is a list of pairs of points."""
        ax.plot(_wfl(_nth(lst, 0)), _hfl(_nth(lst, 1)), color=color, **kwargs)

    def _text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    def _hfl(lst):
        return [a * hf for a in lst]

    def _wfl(lst):
        return [a * wf for a in lst]

    # get number of datasets and strategies:
    n_datasets, n_strategies = scores.shape[0], scores.shape[1]

    # Step 1: rank data: best algorithm gets rank of 1 second best rank of 2...
    # in case of ties average ranks are assigned
    if is_errors:
        # low is good -> rank 1
        ranked_data = rankdata(scores, axis=1)
    else:
        # assign opposite ranks
        ranked_data = rankdata(-1 * scores, axis=1)

    # Step 2: calculate average rank per strategy
    avranks = ranked_data.mean(axis=0)

    # Step 3 : check whether Friedman test is significant
    is_significant = _check_friedman(n_strategies, n_datasets, ranked_data, alpha)

    # Step 4: If Friedman test is significant calculate critical difference as part of
    # Nemenyi post-hoc test

    if is_significant:
        if alpha == 0.01:
            qalpha = [
                0.000,
                2.576,
                2.913,
                3.113,
                3.255,
                3.364,
                3.452,
                3.526,
                3.590,
                3.646,
                3.696,
                3.741,
                3.781,
                3.818,
                3.853,
                3.884,
                3.914,
                3.941,
                3.967,
                3.992,
                4.015,
                4.037,
                4.057,
                4.077,
                4.096,
                4.114,
                4.132,
                4.148,
                4.164,
                4.179,
                4.194,
                4.208,
                4.222,
                4.236,
                4.249,
                4.261,
                4.273,
                4.285,
                4.296,
                4.307,
                4.318,
                4.329,
                4.339,
                4.349,
                4.359,
                4.368,
                4.378,
                4.387,
                4.395,
                4.404,
                4.412,
                4.420,
                4.428,
                4.435,
                4.442,
                4.449,
                4.456,
            ]

        elif alpha == 0.05:
            qalpha = [
                0.000,
                1.960,
                2.344,
                2.569,
                2.728,
                2.850,
                2.948,
                3.031,
                3.102,
                3.164,
                3.219,
                3.268,
                3.313,
                3.354,
                3.391,
                3.426,
                3.458,
                3.489,
                3.517,
                3.544,
                3.569,
                3.593,
                3.616,
                3.637,
                3.658,
                3.678,
                3.696,
                3.714,
                3.732,
                3.749,
                3.765,
                3.780,
                3.795,
                3.810,
                3.824,
                3.837,
                3.850,
                3.863,
                3.876,
                3.888,
                3.899,
                3.911,
                3.922,
                3.933,
                3.943,
                3.954,
                3.964,
                3.973,
                3.983,
                3.992,
                4.001,
                4.009,
                4.017,
                4.025,
                4.032,
                4.040,
                4.046,
            ]
        elif alpha == 0.1:
            qalpha = [
                0.000,
                1.645,
                2.052,
                2.291,
                2.460,
                2.589,
                2.693,
                2.780,
                2.855,
                2.920,
                2.978,
                3.030,
                3.077,
                3.120,
                3.159,
                3.196,
                3.230,
                3.261,
                3.291,
                3.319,
                3.346,
                3.371,
                3.394,
                3.417,
                3.439,
                3.459,
                3.479,
                3.498,
                3.516,
                3.533,
                3.550,
                3.567,
                3.582,
                3.597,
                3.612,
                3.626,
                3.640,
                3.653,
                3.666,
                3.679,
                3.691,
                3.703,
                3.714,
                3.726,
                3.737,
                3.747,
                3.758,
                3.768,
                3.778,
                3.788,
                3.797,
                3.806,
                3.814,
                3.823,
                3.831,
                3.838,
                3.846,
            ]
            #
        else:
            raise Exception("alpha must be 0.01, 0.05 or 0.1")

        if cliques is None:
            # calculate critical difference with Nemenyi
            cd = qalpha[n_strategies] * np.sqrt(
                n_strategies * (n_strategies + 1) / (6 * n_datasets)
            )

            # Step 5: compute statistically similar cliques
            cliques = np.tile(avranks, (n_strategies, 1)) - np.tile(
                np.vstack(avranks.T), (1, n_strategies)
            )
            cliques[cliques < 0] = np.inf
            cliques = cliques < cd

            for i in range(n_strategies - 1, 0, -1):
                if np.all(cliques[i - 1, cliques[i, :]] == cliques[i, cliques[i, :]]):
                    cliques[i, :] = 0

            n = np.sum(cliques, 1)
            cliques = cliques[n > 1, :]

    # If Friedman test is not significant everything has to be one clique
    else:  # Friedman test is not significant
        if cliques is None:
            cliques = [
                [
                    1,
                ]
                * n_strategies
            ]
        else:  # cliques were passed as argument
            if cliques != [
                [
                    1,
                ]
                * n_strategies
            ]:
                raise ValueError(
                    "No significant difference in Friedman test found. "
                    "All strategies have to be in one clique."
                )

    # Step 6 create the diagram:
    # check from where to where the axis has to go
    lowv = min(1, int(math.floor(min(avranks))))
    highv = max(len(avranks), int(math.ceil(max(avranks))))

    # set up the figure
    width = float(width)
    textspace = float(textspace)

    cline = 0.6  # space needed above scale
    linesblank = 0  # lines between scale and text
    scalewidth = width - 2 * textspace

    # calculate height needed height
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((n_strategies + 1) / 2) * 0.2 + minnotsignificant + 0.2

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1.0 / height  # height factor
    wf = 1.0 / width

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # draw scale
    _line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    # add ticks to scale
    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        _line([(_rankpos(a), cline - tick / 2), (_rankpos(a), cline)], linewidth=2)

    for a in range(lowv, highv + 1):
        _text(
            _rankpos(a),
            cline - tick / 2 - 0.05,
            str(a),
            ha="center",
            va="bottom",
            size=16,
        )

    # sort out lines and text based on whether order is reversed or not
    space_between_names = 0.24
    for i in range(math.ceil(len(avranks) / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        if reverse:
            _line(
                [
                    (_rankpos(avranks[i]), cline),
                    (_rankpos(avranks[i]), chei),
                    (textspace + scalewidth + 0.1, chei),
                ],
                linewidth=linewidth,
            )
            _text(
                textspace + scalewidth + 0.2,
                chei,
                labels[i],
                ha="left",
                va="center",
                size=16,
            )
            _text(
                textspace + scalewidth - 0.3,
                chei - 0.075,
                format(avranks[i], ".4f"),
                ha="left",
                va="center",
                size=10,
            )
        else:
            _line(
                [
                    (_rankpos(avranks[i]), cline),
                    (_rankpos(avranks[i]), chei),
                    (textspace - 0.1, chei),
                ],
                linewidth=linewidth,
            )
            _text(
                textspace - 0.2,
                chei,
                labels[i],
                ha="right",
                va="center",
                size=16,
            )
            _text(
                textspace + 0.3,
                chei - 0.075,
                format(avranks[i], ".4f"),
                ha="right",
                va="center",
                size=10,
            )

    for i in range(math.ceil(len(avranks) / 2), len(avranks)):
        chei = cline + minnotsignificant + (len(avranks) - i - 1) * space_between_names
        if reverse:
            _line(
                [
                    (_rankpos(avranks[i]), cline),
                    (_rankpos(avranks[i]), chei),
                    (textspace - 0.1, chei),
                ],
                linewidth=linewidth,
            )
            _text(
                textspace - 0.2,
                chei,
                labels[i],
                ha="right",
                va="center",
                size=16,
            )
            _text(
                textspace + 0.3,
                chei - 0.075,
                format(avranks[i], ".4f"),
                ha="right",
                va="center",
                size=10,
            )
        else:
            _line(
                [
                    (_rankpos(avranks[i]), cline),
                    (_rankpos(avranks[i]), chei),
                    (textspace + scalewidth + 0.1, chei),
                ],
                linewidth=linewidth,
            )
            _text(
                textspace + scalewidth + 0.2,
                chei,
                labels[i],
                ha="left",
                va="center",
                size=16,
            )
            _text(
                textspace + scalewidth - 0.3,
                chei - 0.075,
                format(avranks[i], ".4f"),
                ha="left",
                va="center",
                size=10,
            )

    # draw lines for cliques
    start = cline + 0.2
    side = -0.02
    height = 0.1
    i = 1
    achieved_half = False
    for clq in cliques:
        positions = np.where(np.array(clq) == 1)[0]
        min_idx = np.array(positions).min()
        max_idx = np.array(positions).max()
        if not (min_idx >= len(labels) / 2 and achieved_half):
            start = cline + 0.25
            achieved_half = True
        _line(
            [
                (_rankpos(avranks[min_idx]) - side, start),
                (_rankpos(avranks[max_idx]) + side, start),
            ],
            linewidth=linewidth_sign,
        )
        start += height
    plt.show()
