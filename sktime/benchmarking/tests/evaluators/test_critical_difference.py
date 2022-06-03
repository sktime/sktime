# -*- coding: utf-8 -*-
"""Test for critical difference diagram."""
import random

import matplotlib.pyplot as plt
import numpy as np

from sktime.benchmarking.evaluators.diagrams._critical_difference_diagram import (
    _plot_critical_difference_diagram,
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
