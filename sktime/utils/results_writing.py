# -*- coding: utf-8 -*-
""" function to write results to file.


"""

__author__ = [""]

from sktime.utils.data_io import write_results_to_uea_format

if __name__ == "__main__":
    actual = [1, 1, 2, 2, 1, 1, 2, 2]
    preds = [1, 1, 2, 2, 1, 2, 1, 2]
    probas = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.4, 0.6],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.4, 0.6],
        [0.7, 0.3],
        [0.2, 0.8],
    ]

    write_results_to_uea_format(
        path="../exampleResults",
        strategy_name="dummy_classifier",
        dataset_name="banana_point",
        y_true=actual,
        y_pred=preds,
        split="TEST",
        resample_seed=0,
        y_proba=probas,
        second_line="buildTime=100000,num_dummy_things=2",
    )
