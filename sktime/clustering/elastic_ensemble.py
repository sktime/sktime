# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union, List
from itertools import product

import numpy as np
import time
from numpy.random import RandomState
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import rand_score
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RandomizedSearchCV,
    ShuffleSplit,
    cross_val_predict,
)

from sktime.clustering.base import BaseClusterer, TimeSeriesInstances
from sktime.clustering.metrics.averaging import mean_average
from sktime.distances import distance_factory, pairwise_distance
from sktime.distances._ddtw import average_of_slope_transform
from sktime.clustering.k_means import TimeSeriesKMeans

class ElasticEnsemble(BaseClusterer):

    def __init__(
            self,
            distance_measures: List[str] = "all",
            proportion_of_param_options: float = 1.0,
            proportion_train_in_param_finding: float = 1.0,
            proportion_train_for_test: float = 1.0,
            random_state: int = 1,
            verbose: bool = False,
            majority_vote: bool = False,
            n_clusters: int = 8,
            scoring_method: str = 'rand_score'
    ):
        self.distance_measures = distance_measures
        self.proportion_of_param_options = proportion_of_param_options
        self.proportion_train_in_param_finding = proportion_train_in_param_finding
        self.proportion_train_for_test = proportion_train_for_test
        self.random_state = random_state
        self.verbose = verbose
        self.majority_vote = majority_vote
        self.n_clusters = n_clusters
        self.scoring_method = scoring_method

        if distance_measures == "all":
            self._distance_measures = [
                "dtw",
                "ddtw",
                "wdtw",
                "wddtw",
                "lcss",
                "erp",
                "msm",
            ]
        else:
            self._distance_measures = distance_measures

    def _fit(self, X: TimeSeriesInstances) -> np.ndarray:
        der_X = None
        if 'ddtw' in self.distance_measures or 'wddtw' in self.distance_measures:
            der_X = average_of_slope_transform(X)

        self.train_accs_by_clusterer = np.zeros(len(self.distance_measures))
        self.estimators_ = [None] * len(self.distance_measures)
        self._random_state = check_random_state(self.random_state)

        param_train_x = None
        der_param_train_x = None

        if self.proportion_train_in_param_finding < 1:
            if self.verbose is True:
                print(  # noqa: T001
                    "Restricting training cases for parameter optimisation: ", end=""
                )
            shuffle_split = ShuffleSplit(
                n_splits=1,
                test_size=1 - self.proportion_train_in_param_finding,
                random_state=self._random_state,
            )
            for train_index, _ in shuffle_split.split(X):
                param_train_x = X[train_index, :]
                if der_X is not None:
                    der_param_train_x = der_X[train_index, :]
                if self.verbose is True:
                    print(  # noqa: T001
                        "using "
                        + str(len(param_train_x))
                        + " training cases instead of "
                        + str(len(X))
                        + " for parameter optimisation"
                    )
        else:
            if self.verbose is True:
                print(  # noqa: T001
                    "Using all training cases for parameter optimisation"
                )
            param_train_x = X
            if der_X is not None:
                der_param_train_x = der_X

        self.constituent_build_times = []

        if self.verbose is True:
            print(  # noqa: T001
                "Using " + str(100 * self.proportion_of_param_options) + " parameter "
                                                                         "options per "
                                                                         "measure"
            )

        for dm in range(0, len(self.distance_measures)):
            this_measure = self.distance_measures[dm]

            # uses the appropriate training data as required (either full or
            # smaller sample as per the StratifiedShuffleSplit)
            param_train_to_use = param_train_x
            full_train_to_use = X
            if this_measure == "ddtw" or this_measure == "wddtw":
                param_train_to_use = der_param_train_x
                full_train_to_use = der_X
                if this_measure == "ddtw":
                    this_measure = "dtw"
                elif this_measure == "wddtw":
                    this_measure = "wdtw"

            start_build_time = time.time()
            if self.verbose is True:
                if (
                        self.distance_measures[dm] == "ddtw"
                        or self.distance_measures[dm] == "wddtw"
                ):
                    print(  # noqa: T001
                        "Currently evaluating "
                        + str(self.distance_measures[dm].__name__)
                        + " (implemented as "
                        + str(this_measure.__name__)
                        + " with pre-transformed derivative data)"
                    )
                else:
                    print(  # noqa: T001
                        "Currently evaluating "
                        + str(self.distance_measures[dm].__name__)
                    )

            sklearn_verbose = 0
            if self.verbose is True:
                sklearn_verbose = 1

            if self.proportion_of_param_options == 1:

                grid = GridSearchCV(
                    estimator=TimeSeriesKMeans(
                        metric=this_measure, n_clusters=self.n_clusters
                    ),
                    param_grid=ElasticEnsemble._get_100_param_options(
                        self.distance_measures[dm], X
                    ),
                    cv=LeaveOneOut(),
                    scoring=self.scoring_method,
                    verbose=sklearn_verbose,
                )
                grid.fit(param_train_to_use)

                # Else, used RandomizedSearchCV to randomly sample parameter
                # options for each measure
            else:
                grid = RandomizedSearchCV(
                    estimator=TimeSeriesKMeans(
                        metric=this_measure, n_clusters=self.n_clusters
                    ),
                    param_distributions=ElasticEnsemble._get_100_param_options(
                        self.distance_measures[dm], X
                    ),
                    n_iter=100 * self.proportion_of_param_options,
                    cv=LeaveOneOut(),
                    scoring=self.scoring_method,
                    random_state=self.random_state,
                    verbose=sklearn_verbose,
                )
                grid.fit(param_train_to_use)

            if self.majority_vote:
                acc = 1
            else:
                best_model = TimeSeriesKMeans(
                    n_clusters=self.n_clusters,
                    metric=this_measure,
                    distance_params=grid.best_params_["distance_params"],
                )
                preds = cross_val_predict(
                    best_model, full_train_to_use, cv=LeaveOneOut()
                )
                acc = rand_score(preds)

            if self.verbose is True:
                print(  # noqa: T001
                    "Training rand index for "
                    + str(self.distance_measures[dm].__name__)
                    + ": "
                    + str(acc)
                    + " (with parameter setting: "
                    + str(grid.best_params_["distance_params"])
                    + ")"
                )

            best_model = TimeSeriesKMeans(
                n_clusters=self.n_clusters,
                metric=this_measure,
                distance_params=grid.best_params_["distance_params"],
            )
            best_model.fit(full_train_to_use)
            end_build_time = time.time()

            self.constituent_build_times.append(str(end_build_time - start_build_time))
            self.estimators_[dm] = best_model
            self.train_accs_by_classifier[dm] = acc
        return self

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pass

    @staticmethod
    def _get_100_param_options(distance_measure, train_x=None, data_dim_to_use=0):
        def get_inclusive(min_val, max_val, num_vals):
            inc = (max_val - min_val) / (num_vals - 1)
            return np.arange(min_val, max_val + inc / 2, inc)

        if distance_measure == "dtw" or distance_measure == "ddtw":
            return {"distance_params": [{"window": x / 100} for x in range(0, 100)]}
        elif distance_measure == "wdtw" or distance_measure == "wddtw":
            return {"distance_params": [{"g": x / 100} for x in range(0, 100)]}
        elif distance_measure == "lcss":
            train_std = np.std(train_x)
            epsilons = get_inclusive(train_std * 0.2, train_std, 10)
            deltas = get_inclusive(0, 0.25, 10)
            a = list(product(epsilons, deltas))
            return {
                "distance_params": [
                    {"epsilon": a[x][0], "window": a[x][1]} for x in range(0, len(a))
                ]
            }
        elif distance_measure == "erp":
            train_std = np.std(train_x)
            band_sizes = get_inclusive(0, 0.25, 10)
            g_vals = get_inclusive(train_std * 0.2, train_std, 10)
            b_and_g = list(product(band_sizes, g_vals))
            return {
                "distance_params": [
                    {"window": b_and_g[x][0], "g": b_and_g[x][1]}
                    for x in range(0, len(b_and_g))
                ]
            }
        elif distance_measure == "msm":
            a = get_inclusive(0.01, 0.1, 25)
            b = get_inclusive(0.1, 1, 26)
            c = get_inclusive(1, 10, 26)
            d = get_inclusive(10, 100, 26)
            return {
                "distance_params": [
                    {"c": x} for x in np.concatenate([a, b[1:], c[1:], d[1:]])
                ]
            }
        # elif distance_measure == twe_distance
        else:
            raise NotImplementedError(
                "EE does not currently support: " + str(distance_measure)
            )
