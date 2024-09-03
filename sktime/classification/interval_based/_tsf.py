"""Time Series Forest (TSF) Classifier.

Interval based TSF classifier, extracts basic summary features from random intervals.
"""

__author__ = ["kkoziara", "luiszugasti", "kanand77"]
__all__ = ["TimeSeriesForestClassifier"]

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sktime.base._panel.forest._tsf import BaseTimeSeriesForest, _transform
from sktime.classification.base import BaseClassifier


class TimeSeriesForestClassifier(
    BaseTimeSeriesForest, ForestClassifier, BaseClassifier
):
    """Time series forest classifier.

    A time series forest is an ensemble of decision trees built on random intervals.
    Overview: Input n series length m.
    For each tree

    - sample sqrt(m) intervals,
    - find mean, std and slope for each interval, concatenate to form new
    data set, if inner series length is set, then intervals are sampled
    within bins of length inner_series_length.
    - build decision tree on new data set.

    Ensemble the trees with averaged probability estimates.

    This implementation deviates from the original in minor ways. It samples
    intervals with replacement and does not use the splitting criteria tiny
    refinement described in [1].

    This classifier is intentionally written with low configurability,
    for performance reasons.

    * for a more configurable tree based ensemble,
      use ``sktime.classification.ensemble.ComposableTimeSeriesForestClassifier``,
      which also allows switching the base estimator.
    * to build a a time series forest with configurable ensembling, base estimator,
      and/or feature extraction, fully from composable blocks,
      combine ``sktime.classification.ensemble.BaggingClassifier`` with
      any classifier pipeline, e.g., pipelining any ``sklearn`` classifier
      with any time series feature extraction, e.g., ``Summarizer``

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    min_interval : int, default=3
        Minimum length of an interval.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``-1`` means using all processors.
    inner_series_length: int, default=None
        The maximum length of unique segments within X from which we extract
        intervals is determined. This helps prevent the extraction of
        intervals that span across distinct inner series.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.
    feature_importances_ : pandas Dataframe of shape (series_length, 3)
        The feature temporal importances for each feature type (mean, std, slope).
        It shows how much each time point of your input dataset, through the
        feature types extracted (mean, std, slope), contributed to the predictions.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/TSF.java>`_.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction",Information Sciences, 239, 2013

    Examples
    --------
    >>> from sktime.classification.interval_based import TimeSeriesForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = TimeSeriesForestClassifier(n_estimators=5)
    >>> clf.fit(X_train, y_train)
    TimeSeriesForestClassifier(n_estimators=5)
    >>> y_pred = clf.predict(X_test)
    """

    _feature_types = ["mean", "std", "slope"]
    _base_estimator = DecisionTreeClassifier(criterion="entropy")

    _tags = {
        # packaging info
        # --------------
        "authors": ["kkoziara", "luiszugasti", "kanand77"],
        "maintainers": ["kkoziara", "luiszugasti", "kanand77"],
        "python_dependencies": ["joblib"],
        # estimator type
        # --------------
        "capability:feature_importance": True,
        "capability:predict_proba": True,
    }

    def __init__(
        self,
        min_interval=3,
        n_estimators=200,
        inner_series_length: Optional[int] = None,
        n_jobs=1,
        random_state=None,
    ):
        self.criterion = "gini"  # needed for BaseForest in sklearn > 1.4.0,
        # because sklearn tag logic looks at this attribute

        super().__init__(
            min_interval=min_interval,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            inner_series_length=inner_series_length,
        )
        BaseClassifier.__init__(self)

    def fit(self, X, y, **kwargs):
        """Wrap fit to call BaseClassifier.fit.

        This is a fix to get around the problem with multiple inheritance. The problem
        is that if we just override _fit, this class inherits the fit from the sklearn
        class BaseTimeSeriesForest. This is the simplest solution, albeit a little
        hacky.
        """
        return BaseClassifier.fit(self, X=X, y=y, **kwargs)

    def predict(self, X, **kwargs) -> np.ndarray:
        """Wrap predict to call BaseClassifier.predict."""
        return BaseClassifier.predict(self, X=X, **kwargs)

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        """Wrap predict_proba to call BaseClassifier.predict_proba."""
        return BaseClassifier.predict_proba(self, X=X, **kwargs)

    def _fit(self, X, y):
        BaseTimeSeriesForest._fit(self, X=X, y=y)

    def _predict(self, X) -> np.ndarray:
        """Find predictions for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        proba = self.predict_proba(X)
        return np.asarray([self.classes_[np.argmax(prob)] for prob in proba])

    def _predict_proba(self, X) -> np.ndarray:
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : np.ndarray of shape = (n_instances, n_classes)
            Predicted probabilities
        """
        from joblib import Parallel, delayed

        X = X.squeeze(1)
        y_probas = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_single_classifier_proba)(
                X, self.estimators_[i], self.intervals_[i]
            )
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self.n_estimators
        )
        return output

    def _get_fitted_params(self):
        params = super()._get_fitted_params()
        params.update({"n_classes": self.n_classes_, "fit_time": self.fit_time_})
        return params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10}
        else:
            return {"n_estimators": 2}

    def _extract_feature_importance_by_feature_type_per_tree(
        self, tree_feature_importance: np.array, feature_type: str
    ) -> np.array:
        """Return feature importance.

        Extracting the feature importance corresponding from a feature type
        (eg. "mean", "std", "slope") from tree feature importance

        Parameters
        ----------
        tree_feature_importance : array-like of shape (n_features_in,)
            The feature importance per feature in an estimator, n_intervals x number
            of feature types
        feature_type : str
            feature type belonging to self.feature_types

        Returns
        -------
        self : array-like of shape (n_intervals,)
            Feature importance corresponding from a feature type.
        """
        feature_index = np.argwhere(
            [
                feature_type == feature_type_recorded
                for feature_type_recorded in self._feature_types
            ]
        )[0, 0]

        feature_type_feature_importance = tree_feature_importance[
            [
                interval_index + feature_index
                for interval_index in range(
                    0, len(tree_feature_importance), len(self._feature_types)
                )
            ]
        ]

        return feature_type_feature_importance

    @property
    def feature_importances_(self, **kwargs) -> pd.DataFrame:
        """Return the temporal feature importances.

        There is an implementation of temporal feature importance in
        BaseTimeSeriesForest in sktime.base._panel.forest._composable
        but TimeseriesForestClassifier is inheriting from
        sktime.base._panel.forest._tsf.py
        which does not have feature_importance_.

        Other feature importance methods implementation:
        >>> from sktime.base._panel.forest._composable import BaseTimeSeriesForest

        Returns
        -------
        feature_importances_ : pandas Dataframe of shape (series_length, 3)
            The feature importances for each feature type (mean, std, slope).
        """
        all_importances_per_feature = {
            _feature_type: np.zeros(self.series_length)
            for _feature_type in self._feature_types
        }

        for tree_index in range(self.n_estimators):
            tree = self.estimators_[tree_index]
            tree_importances = tree.feature_importances_
            tree_intervals = self.intervals_[tree_index]
            for feature_type in self._feature_types:
                feature_type_importances = (
                    self._extract_feature_importance_by_feature_type_per_tree(
                        tree_importances, feature_type
                    )
                )
                for interval_index in range(self.n_intervals):
                    interval = tree_intervals[interval_index]
                    all_importances_per_feature[feature_type][
                        interval[0] : interval[1]
                    ] += feature_type_importances[interval_index]

        temporal_feature_importance = (
            pd.DataFrame(all_importances_per_feature)
            / self.n_estimators
            / self.n_intervals
        )
        return temporal_feature_importance


def _predict_single_classifier_proba(X, estimator, intervals):
    """Find probability estimates for each class for all cases in X."""
    Xt = _transform(X, intervals)
    return estimator.predict_proba(Xt)
