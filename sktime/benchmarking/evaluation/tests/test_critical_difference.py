import numpy as np
import random
from sktime.benchmarking.evaluation.utils._critical_difference_diagram import _plot_critical_difference_diagram

def test_critical_difference():

    max_estimators = 20

    figures = []

    for i in range(2, 20):
        estimators = []
        ranks = []
        for j in range(1, i + 1):
            estimators.append(f'cls{j}')
            ranks.append(
                [random.uniform(1.1,len(estimators)) for _ in range(len(estimators))][-1]
            )
        sorted_rank = np.flip(np.sort(ranks))
        figures.append(_plot_critical_difference_diagram(estimators, list(sorted_rank)))

    for fig in figures:
        fig.show()