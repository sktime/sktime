# -*- coding: utf-8 -*-
"""Test scenarios for classification and regression.

Contains TestScenario concrete children to run in tests for classifiers/regressirs.
"""

__author__ = ["fkiraly"]

__all__ = [
    "scenarios_classification",
    "scenarios_early_classification",
    "scenarios_regression",
]

from inspect import isclass

from sktime.base import BaseObject
from sktime.classification.base import BaseClassifier
from sktime.classification.early_classification import BaseEarlyClassifier
from sktime.regression.base import BaseRegressor
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_classification_y, _make_panel_X
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


class ClassifierTestScenario(TestScenario, BaseObject):
    """Generic test scenario for classifiers."""

    def get_args(self, key, obj=None, deepcopy_args=True):
        """Return args for key. Can be overridden for dynamic arg generation.

        If overridden, must not have any side effects on self.args
            e.g., avoid assignments args[key] = x without deepcopying self.args first

        Parameters
        ----------
        key : str, argument key to construct/retrieve args for
        obj : obj, optional, default=None. Object to construct args for.
        deepcopy_args : bool, optional, default=True. Whether to deepcopy return.

        Returns
        -------
        args : argument dict to be used for a method, keyed by `key`
            names for keys need not equal names of methods these are used in
                but scripted method will look at key with same name as default
        """
        # use same args for predict-like functions as for predict
        if key in ["predict_proba", "decision_function"]:
            key = "predict"

        return super(ClassifierTestScenario, self).get_args(
            key=key, obj=obj, deepcopy_args=deepcopy_args
        )

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
        """

        def get_tag(obj, tag_name):
            if isclass(obj):
                return obj.get_class_tag(tag_name)
            else:
                return obj.get_tag(tag_name)

        regr_or_classf = (BaseClassifier, BaseEarlyClassifier, BaseRegressor)

        # applicable only if obj inherits from BaseClassifier, BaseEarlyClassifier or
        #   BaseRegressor. currently we test both classifiers and regressors using these
        #   scenarios
        if not isinstance(obj, regr_or_classf) and not issubclass(obj, regr_or_classf):
            return False

        # if X is multivariate, applicable only if can handle multivariate
        is_multivariate = not self.get_tag("X_univariate")
        if is_multivariate and not get_tag(obj, "capability:multivariate"):
            return False

        # if X is unequal length, applicable only if can handle unequal length
        is_unequal_length = self.get_tag("X_unequal_length")
        if is_unequal_length and not get_tag(obj, "capability:unequal_length"):
            return False

        return True


y = _make_classification_y(n_instances=10, random_state=RAND_SEED)
X = _make_panel_X(n_instances=10, n_timepoints=20, random_state=RAND_SEED, y=y)
X_test = _make_panel_X(n_instances=5, n_timepoints=20, random_state=RAND_SEED)

X_multivariate = _make_panel_X(
    n_instances=10, n_columns=2, n_timepoints=20, random_state=RAND_SEED, y=y
)
X_test_multivariate = _make_panel_X(
    n_instances=5, n_columns=2, n_timepoints=20, random_state=RAND_SEED
)


class ClassifierFitPredict(ClassifierTestScenario):
    """Fit/predict with univariate panel X, nested_univ mtype, and labels y."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": False,
        "is_enabled": True,
        "n_classes": 2,
    }

    args = {
        "fit": {"y": y, "X": X},
        "predict": {"X": X_test},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


y3 = _make_classification_y(n_instances=11, n_classes=3, random_state=RAND_SEED)
X_np = _make_panel_X(
    n_instances=11, n_timepoints=17, random_state=RAND_SEED, y=y3, return_numpy=True
)
X_test_np = _make_panel_X(
    n_instances=6, n_timepoints=17, random_state=RAND_SEED, return_numpy=True
)


class ClassifierFitPredictNumpy(ClassifierTestScenario):
    """Fit/predict with univariate panel X, numpy3D mtype, and labels y."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": False,
        "is_enabled": False,
        "n_classes": 3,
    }

    args = {
        "fit": {"y": y3, "X": X_np},
        "predict": {"X": X_test_np},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


class ClassifierFitPredictMultivariate(ClassifierTestScenario):
    """Fit/predict with multivariate panel X and labels y."""

    _tags = {
        "X_univariate": False,
        "X_unequal_length": False,
        "is_enabled": True,
        "n_classes": 2,
    }

    args = {
        "fit": {"y": y, "X": X_multivariate},
        "predict": {"X": X_test_multivariate},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


X_unequal_length = _make_hierarchical(
    hierarchy_levels=(10,), min_timepoints=10, max_timepoints=15, random_state=RAND_SEED
)
X_unequal_length_test = _make_hierarchical(
    hierarchy_levels=(5,), min_timepoints=10, max_timepoints=15, random_state=RAND_SEED
)


class ClassifierFitPredictUnequalLength(ClassifierTestScenario):
    """Fit/predict with univariate panel X and labels y, unequal length series."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": True,
        "is_enabled": True,
        "n_classes": 2,
    }

    args = {
        "fit": {"y": y, "X": X_unequal_length},
        "predict": {"X": X_unequal_length_test},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


scenarios_classification = [
    ClassifierFitPredict,
    ClassifierFitPredictNumpy,
    ClassifierFitPredictMultivariate,
    ClassifierFitPredictUnequalLength,
]

# same scenarios used for early classification
scenarios_early_classification = [
    ClassifierFitPredict,
    ClassifierFitPredictNumpy,
    ClassifierFitPredictMultivariate,
    ClassifierFitPredictUnequalLength,
]

# we use the same scenarios for regression, as in the old test suite
scenarios_regression = [
    ClassifierFitPredict,
    ClassifierFitPredictNumpy,
    ClassifierFitPredictMultivariate,
    ClassifierFitPredictUnequalLength,
]
