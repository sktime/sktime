# -*- coding: utf-8 -*-
"""Test for critical difference diagram."""
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sktime.benchmarking.evaluation.diagrams._critical_difference_diagram import (
    _plot_critical_difference_diagram,
    create_critical_difference_diagram,
)

DEBUG = False


def test_critical_difference():
    """Test different size critical difference diagrams."""
    random.seed(20)

    min_estimators = 2
    max_estimators = 20

    figures = []

    for i in range(min_estimators, max_estimators):
        estimators = []
        ranks = []
        for j in range(1, i + 1):
            estimators.append(f"cls{j}")
            ranks.append(
                [random.uniform(1.1, len(estimators)) for _ in range(len(estimators))][
                    -1
                ]
            )
        sorted_rank = np.flip(np.sort(ranks))
        figures.append(_plot_critical_difference_diagram(estimators, list(sorted_rank)))

    for fig in figures:
        assert isinstance(fig, plt.Figure)
        if DEBUG is True:
            fig.show()


def test_crit():
    """Test complete critical difference."""
    num_estimators = 10
    datasets = 20
    np.random.seed(20)
    metric1 = np.random.rand(num_estimators * datasets)
    metric2 = np.random.rand(num_estimators * datasets)
    metric3 = np.random.rand(num_estimators * datasets)

    data = []
    for i in range(1, num_estimators + 1):
        curr_estimators = f"cls_{i}"
        for j in range(datasets):
            curr_dataset = f"dataset_{j}"
            metric_pos = i * j
            data.append(
                [
                    curr_estimators,
                    curr_dataset,
                    metric1[metric_pos],
                    metric2[metric_pos],
                    metric3[metric_pos],
                ]
            )

    df = pd.DataFrame(data)

    df = pd.read_csv("./test.csv", index_col=False)
    figure = create_critical_difference_diagram(df, title="Accuracy")
    assert isinstance(figure, plt.Figure)

    if DEBUG is True:
        figure.show()
