# -*- coding: utf-8 -*-
"""The Elastic Ensemble (EE).

An ensemble of elastic nearest neighbour classifiers.
"""

__author__ = "Jason Lines"
__all__ = ["ElasticEnsemble"]

import os
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import class_distribution
from sktime.classification.base import BaseClassifier
from sktime.classification.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesClassifier,
)
from sktime.distances.elastic_cython import ddtw_distance as ddtw_c
from sktime.distances.elastic_cython import dtw_distance as dtw_c
from sktime.distances.elastic_cython import erp_distance as erp_c
from sktime.distances.elastic_cython import lcss_distance as lcss_c
from sktime.distances.elastic_cython import msm_distance as msm_c
from sktime.distances.elastic_cython import wddtw_distance as wddtw_c
from sktime.distances.elastic_cython import wdtw_distance as wdtw_c
from sktime.transformations.panel.summarize import DerivativeSlopeTransformer
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class ElasticEnsemble(BaseClassifier):
    """The Elastic Ensemble (EE).

    EE as described in [1].

    Overview:

    - Input n series length m
    - EE is an ensemble of elastic nearest neighbor classifiers

    Parameters
    ----------
    distance_measures : list of strings, optional (default="all")
      A list of strings identifying which distance measures to include.
    proportion_of_param_options : float, optional (default=1)
      The proportion of the parameter grid space to search optional.
    proportion_train_in_param_finding : float, optional (default=1)
      The proportion of the train set to use in the parameter search optional.
    proportion_train_for_test : float, optional (default=1)
      The proportion of the train set to use in classifying new cases optional.
    n_jobs : int or None, optional (default=None)
      The number of jobs to run in parallel for both `fit` and `predict`.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors.
    random_state : int, default=0
      The random seed.
    verbose : int, default=0
      If ``>0``, then prints out debug information.

    Attributes
    ----------
    estimators_ : list
      A list storing all classifiers
    train_accs_by_classifier : ndarray
      Store the train accuracies of the classifiers
    train_preds_by_classifier : list
      Store the train predictions of each classifier

    Notes
    -----
    ..[1] Jason Lines and Anthony Bagnall,
          "Time Series Classification with Ensembles of Elastic Distance Measures",
              Data Mining and Knowledge Discovery, 29(3), 2015.
    https://link.springer.com/article/10.1007/s10618-014-0361-2

    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        distance_measures="all",
        proportion_of_param_options=1.0,
        proportion_train_in_param_finding=1.0,
        proportion_train_for_test=1.0,
        n_jobs=None,
        random_state=0,
        verbose=0,
    ):
        if distance_measures == "all":
            self.distance_measures = [
                dtw_c,
                ddtw_c,
                wdtw_c,
                wddtw_c,
                lcss_c,
                erp_c,
                msm_c,
            ]
        else:
            self.distance_measures = distance_measures
        self.proportion_train_in_param_finding = proportion_train_in_param_finding
        self.proportion_of_param_options = proportion_of_param_options
        self.proportion_train_for_test = proportion_train_for_test
        self.estimators_ = None
        self.train_accs_by_classifier = None
        self.train_preds_by_classifier = None
        self.classes_ = None
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.train = None
        self.constituent_build_times = None
        super(ElasticEnsemble, self).__init__()

    def fit(self, X, y):
        """Build an ensemble of 1-NN classifiers from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,
            it must have a single column. BOSS not configured
            to handle multivariate
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_pandas=False)

        # Derivative DTW (DDTW) uses the regular DTW algorithm on data that
        # are transformed into derivatives.
        # To increase the efficiency of DDTW we can pre-transform the data
        # into derivatives, and then call the
        # standard DTW algorithm on it, rather than transforming each series
        # every time a distance calculation
        # is made. Please note that using DDTW elsewhere will not benefit
        # from this speed enhancement
        if self.distance_measures.__contains__(
            ddtw_c
        ) or self.distance_measures.__contains__(wddtw_c):
            der_X = DerivativeSlopeTransformer().fit_transform(X)
            # reshape X for use with the efficient cython distance measures
            if isinstance(X, pd.DataFrame):
                der_X = np.array(
                    [np.asarray([x]).reshape(1, len(x)) for x in der_X.iloc[:, 0]]
                )
        else:
            der_X = None

        # reshape X for use with the efficient cython distance measures
        if isinstance(X, pd.DataFrame):
            X = np.array([np.asarray([x]).reshape(1, len(x)) for x in X.iloc[:, 0]])

        self.train_accs_by_classifier = np.zeros(len(self.distance_measures))
        self.train_preds_by_classifier = [None] * len(self.distance_measures)
        self.estimators_ = [None] * len(self.distance_measures)
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        rand = np.random.RandomState(self.random_state)

        # The default EE uses all training instances for setting parameters,
        # and 100 parameter options per
        # elastic measure. The prop_train_in_param_finding and
        # prop_of_param_options attributes of this class
        # can be used to control this however, using less cases to optimise
        # parameters on the training data
        # and/or using less parameter options.
        #
        # For using less training instances the appropriate number of cases
        # must be sampled from the data.
        # This is achieved through the use of a deterministic
        # StratifiedShuffleSplit
        #
        # For using less parameter options a RandomizedSearchCV is used in
        # place of a GridSearchCV

        param_train_x = None
        der_param_train_x = None
        param_train_y = None

        # If using less cases for parameter optimisation, use the
        # StratifiedShuffleSplit:
        if self.proportion_train_in_param_finding < 1:
            if self.verbose > 0:
                print(  # noqa: T001
                    "Restricting training cases for parameter optimisation: ", end=""
                )
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 - self.proportion_train_in_param_finding,
                random_state=rand,
            )
            for train_index, _ in sss.split(X, y):
                param_train_x = X[train_index, :]
                param_train_y = y[train_index]
                if der_X is not None:
                    der_param_train_x = der_X[train_index, :]
                if self.verbose > 0:
                    print(  # noqa: T001
                        "using "
                        + str(len(param_train_x))
                        + " training cases instead of "
                        + str(len(X))
                        + " for parameter optimisation"
                    )
        # else, use the full training data for optimising parameters
        else:
            if self.verbose > 0:
                print(  # noqa: T001
                    "Using all training cases for parameter optimisation"
                )
            param_train_x = X
            param_train_y = y
            if der_X is not None:
                der_param_train_x = der_X

        self.constituent_build_times = []

        if self.verbose > 0:
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
            if this_measure is ddtw_c or dm is wddtw_c:
                param_train_to_use = der_param_train_x
                full_train_to_use = der_X
                if this_measure is ddtw_c:
                    this_measure = dtw_c
                elif this_measure is wddtw_c:
                    this_measure = wdtw_c

            start_build_time = time.time()
            if self.verbose > 0:
                if (
                    self.distance_measures[dm] is ddtw_c
                    or self.distance_measures[dm] is wddtw_c
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

            # If 100 parameter options are being considered per measure,
            # use a GridSearchCV
            if self.proportion_of_param_options == 1:

                grid = GridSearchCV(
                    estimator=KNeighborsTimeSeriesClassifier(
                        distance=this_measure, n_neighbors=1
                    ),
                    param_grid=ElasticEnsemble._get_100_param_options(
                        self.distance_measures[dm], X
                    ),
                    cv=LeaveOneOut(),
                    scoring="accuracy",
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
                grid.fit(param_train_to_use, param_train_y)

            # Else, used RandomizedSearchCV to randomly sample parameter
            # options for each measure
            else:
                grid = RandomizedSearchCV(
                    estimator=KNeighborsTimeSeriesClassifier(
                        distance=this_measure, n_neighbors=1
                    ),
                    param_distributions=ElasticEnsemble._get_100_param_options(
                        self.distance_measures[dm], X
                    ),
                    n_iter=100 * self.proportion_of_param_options,
                    cv=LeaveOneOut(),
                    scoring="accuracy",
                    n_jobs=self.n_jobs,
                    random_state=rand,
                    verbose=self.verbose,
                )
                grid.fit(param_train_to_use, param_train_y)

            # once the best parameter option has been estimated on the
            # training data, perform a final pass with this parameter option
            # to get the individual predictions with cross_cal_predict (
            # Note: optimisation potentially possible here if a GridSearchCV
            # was used previously. TO-DO: determine how to extract
            # predictions for the best param option from GridSearchCV)
            best_model = KNeighborsTimeSeriesClassifier(
                n_neighbors=1,
                distance=this_measure,
                distance_params=grid.best_params_["distance_params"],
            )
            preds = cross_val_predict(
                best_model, full_train_to_use, y, cv=LeaveOneOut()
            )
            acc = accuracy_score(y, preds)

            if self.verbose > 0:
                print(  # noqa: T001
                    "Training accuracy for "
                    + str(self.distance_measures[dm].__name__)
                    + ": "
                    + str(acc)
                    + " (with parameter setting: "
                    + str(grid.best_params_["distance_params"])
                    + ")"
                )

            # Finally, reset the classifier for this measure and parameter
            # option, ready to be called for test classification
            best_model = KNeighborsTimeSeriesClassifier(
                n_neighbors=1,
                distance=this_measure,
                distance_params=grid.best_params_["distance_params"],
            )
            best_model.fit(full_train_to_use, y)
            end_build_time = time.time()

            self.constituent_build_times.append(str(end_build_time - start_build_time))
            self.estimators_[dm] = best_model
            self.train_accs_by_classifier[dm] = acc
            self.train_preds_by_classifier[dm] = preds

        self._is_fitted = True
        return self

    def predict_proba(self, X):
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]

        Returns
        -------
        array of shape [n, self.n_classes]
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_pandas=False)

        # Derivative DTW (DDTW) uses the regular DTW algorithm on data that
        # are transformed into derivatives.
        # To increase the efficiency of DDTW we can pre-transform the data
        # into derivatives, and then call the
        # standard DTW algorithm on it, rather than transforming each series
        # every time a distance calculation
        # is made. Please note that using DDTW elsewhere will not benefit
        # from this speed enhancement
        if self.distance_measures.__contains__(
            ddtw_c
        ) or self.distance_measures.__contains__(wddtw_c):
            der_X = DerivativeSlopeTransformer().fit_transform(X)
            if isinstance(X, pd.DataFrame):
                der_X = np.array(
                    [np.asarray([x]).reshape(1, len(x)) for x in der_X.iloc[:, 0]]
                )
        else:
            der_X = None

        # reshape X for use with the efficient cython distance measures
        if isinstance(X, pd.DataFrame):
            X = np.array([np.asarray([x]).reshape(1, len(x)) for x in X.iloc[:, 0]])

        output_probas = []
        train_sum = 0

        for c in range(0, len(self.estimators_)):
            if (
                self.distance_measures[c] == ddtw_c
                or self.distance_measures[c] == wddtw_c
            ):
                test_X_to_use = der_X
            else:
                test_X_to_use = X
            this_train_acc = self.train_accs_by_classifier[c]
            this_probas = np.multiply(
                self.estimators_[c].predict_proba(test_X_to_use), this_train_acc
            )
            output_probas.append(this_probas)
            train_sum += this_train_acc

        output_probas = np.sum(output_probas, axis=0)
        output_probas = np.divide(output_probas, train_sum)
        return output_probas

    def predict(self, X, return_preds_and_probas=False):
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]
        return_preds_and_probas: boolean option to return predictions
        Returns
        -------
        array of shape [n, 1]
        """
        probas = self.predict_proba(X)  # does derivative transform within (if required)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        if return_preds_and_probas is False:
            return preds
        else:
            return preds, probas

    def get_train_probs(self, X=None):
        """Find and returns the probability estimates for data X."""
        num_cases = len(self.train_preds_by_classifier[0])
        num_classes = len(self.classes_)
        num_estimators = len(self.estimators_)

        probs = np.zeros((num_cases, num_classes))

        map = LabelEncoder().fit(self.classes_)
        weight_sum = np.sum(self.train_accs_by_classifier)

        for i in range(num_cases):
            for e in range(num_estimators):
                pred_class = map.transform([self.train_preds_by_classifier[e][i]])[0]
                probs[i][pred_class] += self.train_accs_by_classifier[e] / weight_sum
        return probs

    def get_metric_params(self):
        """Return the parameters for the distance metrics used."""
        return {
            self.distance_measures[dm].__name__: str(self.estimators_[dm].metric_params)
            for dm in range(len(self.estimators_))
        }

    def write_constituent_train_files(self, output_file_path, dataset_name, actual_y):
        """Write the train information to file in UEA format."""
        for c in range(len(self.estimators_)):
            measure_name = self.distance_measures[c].__name__

            try:
                os.makedirs(
                    str(output_file_path)
                    + "/"
                    + str(measure_name)
                    + "/Predictions/"
                    + str(dataset_name)
                    + "/"
                )
            except os.error:
                pass  # raises os.error if path already exists

            file = open(
                str(output_file_path)
                + "/"
                + str(measure_name)
                + "/Predictions/"
                + str(dataset_name)
                + "/trainFold"
                + str(self.random_state)
                + ".csv",
                "w",
            )

            # the first line of the output file is in the form of:
            # <classifierName>,<datasetName>,<train/test>
            file.write(str(measure_name) + "," + str(dataset_name) + ",train\n")

            # the second line of the output is free form and
            # classifier-specific; usually this will record info
            # such as build time, paramater options used, any constituent
            # model names for ensembles, etc.
            # file.write(str(self.estimators_[c].best_params_[
            # 'metric_params'])+"\n")
            self.proportion_train_in_param_finding
            file.write(
                str(self.estimators_[c].metric_params)
                + ",build_time,"
                + str(self.constituent_build_times[c])
                + ",prop_of_param_options,"
                + str(self.proportion_of_param_options)
                + ",prop_train_in_param_finding,"
                + str(self.proportion_train_in_param_finding)
                + "\n"
            )

            # third line is training acc
            file.write(str(self.train_accs_by_classifier[c]) + "\n")

            for i in range(len(actual_y)):
                file.write(
                    str(actual_y[i])
                    + ","
                    + str(self.train_preds_by_classifier[c][i])
                    + "\n"
                )
            # preds would go here once stored as part of fit

            file.close()

    @staticmethod
    def _get_100_param_options(distance_measure, train_x=None, data_dim_to_use=0):
        def get_inclusive(min_val, max_val, num_vals):
            inc = (max_val - min_val) / (num_vals - 1)
            return np.arange(min_val, max_val + inc / 2, inc)

        if distance_measure == dtw_c or distance_measure == ddtw_c:
            return {"distance_params": [{"w": x / 100} for x in range(0, 100)]}
        elif distance_measure == wdtw_c or distance_measure == wddtw_c:
            return {"distance_params": [{"g": x / 100} for x in range(0, 100)]}
        elif distance_measure == lcss_c:
            train_std = np.std(train_x)
            epsilons = get_inclusive(train_std * 0.2, train_std, 10)
            deltas = get_inclusive(int(len(train_x[0]) / 4), len(train_x[0]), 10)
            deltas = [int(d) for d in deltas]
            a = list(product(epsilons, deltas))
            return {
                "distance_params": [
                    {"epsilon": a[x][0], "delta": a[x][1]} for x in range(0, len(a))
                ]
            }
        elif distance_measure == erp_c:
            train_std = np.std(train_x)
            band_sizes = get_inclusive(0, 0.25, 10)
            g_vals = get_inclusive(train_std * 0.2, train_std, 10)
            b_and_g = list(product(band_sizes, g_vals))
            return {
                "distance_params": [
                    {"band_size": b_and_g[x][0], "g": b_and_g[x][1]}
                    for x in range(0, len(b_and_g))
                ]
            }
        elif distance_measure == msm_c:
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
