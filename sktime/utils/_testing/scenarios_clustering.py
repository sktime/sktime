"""Test scenarios for clustering.

Contains TestScenario concrete children to run in tests for clusterers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_clustering"]

from inspect import isclass

from sktime.base import BaseObject
from sktime.registry import is_scitype
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import make_clustering_problem
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


# no logic in scenario classes, but placeholder and for pattern homogeneity
class ClustererTestScenario(TestScenario, BaseObject):
    """Generic test scenario for clusterers."""

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
        if key in ["predict_proba"]:
            key = "predict"

        return super().get_args(key=key, obj=obj, deepcopy_args=deepcopy_args)

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

        # applicable only if obj inherits from BaseClusterer
        if not is_scitype(obj, "clusterer"):
            return False

        # if X is multivariate, applicable only if can handle multivariate
        is_multivariate = not self.get_tag("X_univariate")
        if is_multivariate and not get_tag(obj, "capability:multivariate"):
            return False

        # if X is unequal length, applicable only if can handle unequal length
        is_unequal_length = self.get_tag("X_unequal_length")
        if is_unequal_length and not get_tag(obj, "capability:unequal_length"):
            return False

        # if X is out of sample, applicable only if can handle out of sample
        X_out_of_sample = self.get_tag("X_out_of_sample")
        if X_out_of_sample and not get_tag(obj, "capability:out_of_sample"):
            return False

        return True


class ClustererFitPredict(ClustererTestScenario):
    """Fit/predict with panel X, with X in fit same as predict."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": False,
        "X_out_of_sample": False,
        "is_enabled": True,
    }

    @property
    def args(self):
        return {
            "fit": {"X": make_clustering_problem(random_state=RAND_SEED)},
            "predict": {"X": make_clustering_problem(random_state=RAND_SEED)},
        }

    default_method_sequence = ["fit", "predict"]


class ClustererFitPredictOutOfSample(ClustererTestScenario):
    """Fit/predict with panel X, with X in predict different from fit."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": False,
        "X_out_of_sample": True,
        "is_enabled": True,
    }

    @property
    def args(self):
        return {
            "fit": {"X": make_clustering_problem(random_state=RAND_SEED)},
            "predict": {"X": make_clustering_problem(random_state=RAND_SEED + 1)},
        }

    default_method_sequence = ["fit", "predict"]


class ClustererFitPredictUnequalLength(ClustererTestScenario):
    """Fit/predict with univariate panel X, unequal length series."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": True,
        "X_out_of_sample": False,
        "is_enabled": True,
    }

    @property
    def args(self):
        X_unequal_length = _make_hierarchical(
            hierarchy_levels=(10,),
            min_timepoints=10,
            max_timepoints=15,
            random_state=RAND_SEED,
        )
        return {"fit": {"X": X_unequal_length}, "predict": {"X": X_unequal_length}}

    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


class ClustererFitPredictUnequalLengthOutOfSample(ClustererTestScenario):
    """Fit/predict with univariate panel X, unequal length series, out of sample."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": True,
        "X_out_of_sample": True,
        "is_enabled": True,
    }

    @property
    def args(self):
        X_unequal_length = _make_hierarchical(
            hierarchy_levels=(10,),
            min_timepoints=10,
            max_timepoints=15,
            random_state=RAND_SEED,
        )
        X_unequal_length_test = _make_hierarchical(
            hierarchy_levels=(5,),
            min_timepoints=10,
            max_timepoints=15,
            random_state=RAND_SEED,
        )
        return {
            "fit": {"X": X_unequal_length},
            "predict": {"X": X_unequal_length_test},
        }

    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


class ClustererFitPredictMultivariate(ClustererTestScenario):
    """Fit/predict with multivariate panel X."""

    _tags = {
        "X_univariate": False,
        "X_unequal_length": False,
        "X_out_of_sample": False,
        "is_enabled": True,
    }

    @property
    def args(self):
        X_multivariate = make_clustering_problem(
            n_instances=10, n_columns=2, n_timepoints=20, random_state=RAND_SEED
        )
        return {"fit": {"X": X_multivariate}, "predict": {"X": X_multivariate}}

    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


scenarios_clustering = [
    ClustererFitPredict,
    ClustererFitPredictOutOfSample,
    ClustererFitPredictMultivariate,
    ClustererFitPredictUnequalLength,
    ClustererFitPredictUnequalLengthOutOfSample,
]
