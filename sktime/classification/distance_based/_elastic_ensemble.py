"""The Elastic Ensemble (EE).

An ensemble of elastic nearest neighbour classifiers.
"""

__author__ = ["jasonlines", "TonyBagnall"]
__all__ = ["ElasticEnsemble"]

import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    cross_val_predict,
)

from sktime.classification.base import BaseClassifier
from sktime.classification.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesClassifier,
)
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.transformations.panel.summarize import DerivativeSlopeTransformer


class ElasticEnsemble(BaseClassifier):
    """The Elastic Ensemble (EE).

    EE as described in [1].

    Overview:

    - Input n series length m
    - EE is an ensemble of elastic nearest neighbor classifiers

    Parameters
    ----------
    distance_measures : list of strings, optional (default="all")
      A list of strings identifying which distance measures to include. Valid values
      are one or more of: euclidean, dtw, wdtw, ddtw, dwdtw, lcss, erp, msm
    proportion_of_param_options : float, optional (default=1)
      The proportion of the parameter grid space to search optional.
    proportion_train_in_param_finding : float, optional (default=1)
      The proportion of the train set to use in the parameter search optional.
    proportion_train_for_test : float, optional (default=1)
      The proportion of the train set to use in classifying new cases optional.
    n_jobs : int, optional (default=1)
      The number of jobs to run in parallel for both `fit` and `predict`.
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

    Notes
    -----
    ..[1] Jason Lines and Anthony Bagnall,
          "Time Series Classification with Ensembles of Elastic Distance Measures",
              Data Mining and Knowledge Discovery, 29(3), 2015.
    https://link.springer.com/article/10.1007/s10618-014-0361-2

    Examples
    --------
    >>> from sktime.classification.distance_based import ElasticEnsemble
    >>> from sktime.datasets import load_unit_test  # doctest: +SKIP
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test")  # doctest: +SKIP
    >>> clf = ElasticEnsemble(
    ...     proportion_of_param_options=0.1,
    ...     proportion_train_for_test=0.1,
    ...     distance_measures = ["dtw","ddtw"],
    ...     majority_vote=True,
    ... )  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    ElasticEnsemble(...)
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "distance",
    }

    def __init__(
        self,
        distance_measures="all",
        proportion_of_param_options=1.0,
        proportion_train_in_param_finding=1.0,
        proportion_train_for_test=1.0,
        n_jobs=1,
        random_state=0,
        verbose=0,
        majority_vote=False,
    ):
        if distance_measures == "all":
            self.distance_measures = [
                "dtw",
                "ddtw",
                "wdtw",
                "wddtw",
                "lcss",
                "erp",
                "msm",
            ]
        else:
            self.distance_measures = distance_measures
        self.proportion_train_in_param_finding = proportion_train_in_param_finding
        self.proportion_of_param_options = proportion_of_param_options
        self.proportion_train_for_test = proportion_train_for_test
        self.majority_vote = majority_vote
        self.estimators_ = None
        self.train_accs_by_classifier = None
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.train = None
        self.constituent_build_times = None

        super().__init__()

    def _fit(self, X, y):
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
        # Derivative DTW (DDTW) uses the regular DTW algorithm on data that
        # are transformed into derivatives.
        # To increase the efficiency of DDTW we can pre-transform the data
        # into derivatives, and then call the
        # standard DTW algorithm on it, rather than transforming each series
        # every time a distance calculation
        # is made. Please note that using DDTW elsewhere will not benefit
        # from this speed enhancement
        if self.distance_measures.__contains__(
            "ddtw"
        ) or self.distance_measures.__contains__("wddtw"):
            der_X = DerivativeSlopeTransformer().fit_transform(X)
            # convert back to numpy
            if isinstance(der_X, pd.DataFrame):
                der_X = from_nested_to_3d_numpy(der_X)
        else:
            der_X = None

        self.train_accs_by_classifier = np.zeros(len(self.distance_measures))
        self.estimators_ = [None] * len(self.distance_measures)
        rand = np.random.RandomState(self.random_state)

        # The default EE uses all training instances for setting parameters,
        # and 100 parameter options per elastic measure. The
        # prop_train_in_param_finding and prop_of_param_options attributes of this class
        # can be used to control this however, using fewer cases to optimise
        # parameters on the training data and/or using less parameter options.
        #
        # For using fewer training instances the appropriate number of cases must be
        # sampled from the data. This is achieved through the use of a deterministic
        # StratifiedShuffleSplit
        #
        # For using fewer parameter options a RandomizedSearchCV is used in
        # place of a GridSearchCV

        param_train_x = None
        der_param_train_x = None
        param_train_y = None

        # If using less cases for parameter optimisation, use the
        # StratifiedShuffleSplit:
        if self.proportion_train_in_param_finding < 1:
            if self.verbose > 0:
                print(  # noqa: T201
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
                    print(  # noqa: T201
                        "using "
                        + str(len(param_train_x))
                        + " training cases instead of "
                        + str(len(X))
                        + " for parameter optimisation"
                    )
        # else, use the full training data for optimising parameters
        else:
            if self.verbose > 0:
                print(  # noqa: T201
                    "Using all training cases for parameter optimisation"
                )
            param_train_x = X
            param_train_y = y
            if der_X is not None:
                der_param_train_x = der_X

        self.constituent_build_times = []

        if self.verbose > 0:
            print(  # noqa: T201
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
            if self.verbose > 0:
                if (
                    self.distance_measures[dm] == "ddtw"
                    or self.distance_measures[dm] == "wddtw"
                ):
                    print(  # noqa: T201
                        "Currently evaluating "
                        + str(self.distance_measures[dm].__name__)
                        + " (implemented as "
                        + str(this_measure.__name__)
                        + " with pre-transformed derivative data)"
                    )
                else:
                    print(  # noqa: T201
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
                    n_jobs=self._threads_to_use,
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
                    n_iter=int(100 * self.proportion_of_param_options),
                    cv=LeaveOneOut(),
                    scoring="accuracy",
                    n_jobs=self._threads_to_use,
                    random_state=rand,
                    verbose=self.verbose,
                )
                grid.fit(param_train_to_use, param_train_y)

            if self.majority_vote:
                acc = 1
            # once the best parameter option has been estimated on the
            # training data, perform a final pass with this parameter option
            # to get the individual predictions with cross_cal_predict (
            # Note: optimisation potentially possible here if a GridSearchCV
            # was used previously. TO-DO: determine how to extract
            # predictions for the best param option from GridSearchCV)
            else:
                best_model = KNeighborsTimeSeriesClassifier(
                    n_neighbors=1,
                    distance=this_measure,
                    distance_params=grid.best_params_["distance_params"],
                    n_jobs=self._threads_to_use,
                )
                preds = cross_val_predict(
                    best_model, full_train_to_use, y, cv=LeaveOneOut()
                )
                acc = accuracy_score(y, preds)

            if self.verbose > 0:
                print(  # noqa: T201
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
        return self

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]

        Returns
        -------
        array of shape [n, self.n_classes]
        """
        # Derivative DTW (DDTW) uses the regular DTW algorithm on data that
        # are transformed into derivatives.
        # To increase the efficiency of DDTW we can pre-transform the data
        # into derivatives, and then call the
        # standard DTW algorithm on it, rather than transforming each series
        # every time a distance calculation
        # is made. Please note that using DDTW elsewhere will not benefit
        # from this speed enhancement
        if self.distance_measures.__contains__(
            "ddtw"
        ) or self.distance_measures.__contains__("wddtw"):
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
                self.distance_measures[c] == "ddtw"
                or self.distance_measures[c] == "wddtw"
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

    def _predict(self, X, return_preds_and_probas=False) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]
        return_preds_and_probas: boolean option to return predictions
        Returns
        -------
        array of shape [n, 1]
        """
        probas = self._predict_proba(X)  # does derivative transform within if required
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        if return_preds_and_probas is False:
            return preds
        else:
            return preds, probas

    def get_metric_params(self):
        """Return the parameters for the distance metrics used."""
        return {
            self.distance_measures[dm].__name__: str(self.estimators_[dm].metric_params)
            for dm in range(len(self.estimators_))
        }

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {
                "proportion_of_param_options": 0.1,
                "proportion_train_for_test": 0.1,
                "majority_vote": True,
                "distance_measures": ["dtw", "ddtw", "wdtw"],
            }
        else:
            return {
                "proportion_of_param_options": 0.01,
                "proportion_train_for_test": 0.1,
                "majority_vote": True,
                "distance_measures": ["dtw"],
            }
