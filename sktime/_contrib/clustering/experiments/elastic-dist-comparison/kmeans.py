#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import multiprocessing


import numpy as np
import pandas as pd

from sktime._contrib.clustering.experiments.base import BaseExperiment
from sktime._contrib.clustering.experiments.dataset_lists import EQUAL_LENGTH_LOWER
from sktime.benchmarking.experiments import run_clustering_experiment
from sktime.clustering.k_means import TimeSeriesKMeans

ignore_dataset = []


class KmeansExperiment(BaseExperiment):

    @staticmethod
    def _run_experiment_for_dataset(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        dataset_name: str,
        result_path: str,
        experiment_name: str
    ):
        if dataset_name.lower() not in EQUAL_LENGTH_LOWER:
            return

        n_classes = len(set(y_train))

        # k_means_clusterer = TimeSeriesKMeans(
        #     n_clusters=n_classes,
        #     metric="msm",
        #     averaging_method='dba',
        #     average_params={'averaging_distance_metric': 'msm', 'medoids_distance_metric': 'msm'}
        # )

        k_means_clusterer = TimeSeriesKMeans(
            n_clusters=n_classes,
            metric="euclidean",
        )

        run_clustering_experiment(
            X_train,
            k_means_clusterer,
            results_path=f"{result_path}/{experiment_name}",
            trainY=y_train,
            testX=X_test,
            testY=y_test,
            cls_name="kmeans",
            dataset_name=dataset_name,
            resample_id=0,
            overwrite=False,
        )

if __name__ == "__main__":

    server = True
    if server is False:
        dataset_path = os.path.abspath(
                "C:/Users/chris/Documents/Masters/datasets/Univariate_ts/"
        )
        result_path=os.path.abspath("C:/Users/chris/Documents/Masters/results/")
    else:
        dataset_path=os.path.abspath(
            "/root/datasets/Univariate_ts/"
        )
        result_path=os.path.abspath("/root/results/")

    kmeans_experiment = KmeansExperiment(
        experiment_name="msm_dba",
        dataset_path=dataset_path,
        result_path=result_path,
        n_threads=multiprocessing.cpu_count()
    )
    kmeans_experiment.run_experiment()
