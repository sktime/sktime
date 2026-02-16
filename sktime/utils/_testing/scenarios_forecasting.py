"""Test scenarios for forecasters.

Contains TestScenario concrete children to run in tests for forecasters.
"""

__author__ = ["fkiraly"]

__all__ = [
    "forecasting_scenarios_simple",
    "forecasting_scenarios_extended",
    "scenarios_forecasting",
]


from inspect import isclass

import pandas as pd

from sktime.base import BaseObject
from sktime.datatypes import mtype_to_scitype
from sktime.registry import is_scitype
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_panel_X
from sktime.utils._testing.scenarios import TestScenario
from sktime.utils._testing.series import _make_series

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


class ForecasterTestScenario(TestScenario, BaseObject):
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

        # applicable only if obj inherits from BaseForecaster
        if not is_scitype(obj, "forecaster"):
            return False

        # applicable only if number of variables in y complies with scitype:y
        # only rule: multivariate forecasters cannot deal with univariate data
        # univariate forecasters can deal with multivariate data by vectorization
        is_univariate = self.get_tag("univariate_y")

        if is_univariate and get_tag(obj, "scitype:y") == "multivariate":
            return False

        # applicable only if fh is not passed later than it needs to be
        fh_in_fit = self.get_tag("fh_passed_in_fit")

        if not fh_in_fit and get_tag(obj, "requires-fh-in-fit"):
            return False

        # run Panel/Hierarchical scenarios for genuinely Panel/Hierarchical forecasters
        y_scitype = self.get_tag("y_scitype", "Series", raise_error=False)
        scenario_is_hierarchical = y_scitype in ["Panel", "Hierarchical"]

        obj_y_inner_types = get_tag(obj, "y_inner_mtype")
        obj_scitypes = mtype_to_scitype(obj_y_inner_types)
        obj_is_hierarchical = "Panel" in obj_scitypes or "Hierarchical" in obj_scitypes

        # if scenario is hierarchical and obj is not genuinely hierarchical,
        # this would trigger generic vectorization, which is tested in test_base
        if scenario_is_hierarchical and not obj_is_hierarchical:
            return False

        return True

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
        PREDICT_LIKE_FUNCTIONS = ["predict", "predict_var", "predict_proba"]
        # use same args for predict-like functions as for predict
        if key in PREDICT_LIKE_FUNCTIONS:
            key = "predict"

        return super().get_args(key=key, obj=obj, deepcopy_args=deepcopy_args)


class ForecasterFitPredictUnivariateNoX(ForecasterTestScenario):
    """Fit/predict only, univariate y, no X, fh passed late in predict."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True, "is_enabled": False}

    @property
    def args(self):
        return {
            "fit": {
                "y": _make_series(n_timepoints=20, random_state=RAND_SEED),
                "fh": 1,
            },
            "predict": {"fh": 1},
        }

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateNoXEarlyFh(ForecasterTestScenario):
    """Fit/predict only, univariate y, no X, fh passed late in predict."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True}

    @property
    def args(self):
        return {
            "fit": {
                "y": _make_series(n_timepoints=20, random_state=RAND_SEED),
                "fh": 1,
            },
            "predict": {},
        }

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateNoXLateFh(ForecasterTestScenario):
    """Fit/predict only, univariate y, no X, fh passed late in predict."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": False}

    @property
    def args(self):
        return {
            "fit": {"y": _make_series(n_timepoints=20, random_state=RAND_SEED)},
            "predict": {"fh": 1},
        }

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateNoXLongFh(ForecasterTestScenario):
    """Fit/predict only, univariate y, no X, longer fh, passed early in fit."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True, "is_enabled": True}

    @property
    def args(self):
        y_with_name = _make_series(n_timepoints=20, random_state=RAND_SEED)
        y_with_name.name = "foo"
        return {
            "fit": {"y": y_with_name, "fh": [1, 2, 3]},
            "predict": {},
        }

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateWithX(ForecasterTestScenario):
    """Fit/predict only, univariate y, with X, fh passed early in fit."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True, "is_enabled": True}

    @property
    def args(self):
        y_series = _make_series(n_timepoints=20, random_state=RAND_SEED)
        y = pd.DataFrame(y_series, columns=["foo"])

        LONG_X = _make_series(n_columns=2, n_timepoints=30, random_state=RAND_SEED)
        X = LONG_X.iloc[0:20]
        X_test_short = LONG_X.iloc[20:21]

        return {"fit": {"y": y, "X": X, "fh": 1}, "predict": {"X": X_test_short}}

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateWithXLongFh(ForecasterTestScenario):
    """Fit/predict only, univariate y, with X, and longer fh, passed early in fit."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True}

    @property
    def args(self):
        y = _make_series(n_timepoints=20, random_state=RAND_SEED)

        LONG_X = _make_series(n_columns=2, n_timepoints=30, random_state=RAND_SEED)
        X = LONG_X.iloc[0:20]
        X_test = LONG_X.iloc[20:23]

        return {"fit": {"y": y, "X": X, "fh": [1, 2, 3]}, "predict": {"X": X_test}}

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictMultivariateNoX(ForecasterTestScenario):
    """Fit/predict only, multivariate y, no X, fh passed early in fit."""

    _tags = {"univariate_y": False, "fh_passed_in_fit": True, "is_enabled": True}

    @property
    def args(self):
        return {
            "fit": {
                "y": _make_series(n_timepoints=20, n_columns=2, random_state=RAND_SEED),
                "fh": 1,
            },
            "predict": {},
        }

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictMultivariateWithX(ForecasterTestScenario):
    """Fit/predict only, multivariate y, with X, and longer fh, passed early in fit."""

    _tags = {"univariate_y": False, "fh_passed_in_fit": True}

    @property
    def args(self):
        LONG_X = _make_series(n_columns=2, n_timepoints=30, random_state=RAND_SEED)
        X = LONG_X.iloc[0:20]
        X_test = LONG_X.iloc[20:23]

        return {
            "fit": {
                "y": _make_series(n_timepoints=20, n_columns=2, random_state=RAND_SEED),
                "X": X,
                "fh": [1, 2, 3],
            },
            "predict": {"X": X_test},
        }

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictPanelSimple(ForecasterTestScenario):
    """Fit/predict only, univariate Panel y, no X, and longer fh passed early in fit."""

    _tags = {
        "univariate_y": True,
        "fh_passed_in_fit": True,
        "y_scitype": "Panel",
        "is_enabled": True,
    }

    @property
    def args(self):
        y_panel = _make_panel_X(
            n_instances=3,
            n_timepoints=10,
            n_columns=1,
            random_state=RAND_SEED,
            all_positive=True,
        )
        return {"fit": {"y": y_panel, "fh": [1, 2, 3]}, "predict": {}}

    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictHierarchicalSimple(ForecasterTestScenario):
    """Fit/predict only, univariate Hierarchical y, no X, and longer fh in fit."""

    _tags = {
        "univariate_y": True,
        "fh_passed_in_fit": True,
        "y_scitype": "Hierarchical",
        "is_enabled": True,
    }

    @property
    def args(self):
        y_hierarchical = _make_hierarchical(
            hierarchy_levels=(2, 2), n_columns=1, random_state=RAND_SEED
        )

        return {"fit": {"y": y_hierarchical, "fh": [1, 2, 3]}, "predict": {}}

    default_method_sequence = ["fit", "predict"]


forecasting_scenarios_simple = [
    ForecasterFitPredictUnivariateNoX,
    ForecasterFitPredictMultivariateWithX,
]

forecasting_scenarios_extended = [
    ForecasterFitPredictUnivariateNoX,
    ForecasterFitPredictUnivariateNoXEarlyFh,
    ForecasterFitPredictUnivariateNoXLateFh,
    ForecasterFitPredictUnivariateWithX,
    ForecasterFitPredictUnivariateWithXLongFh,
    ForecasterFitPredictMultivariateNoX,
    ForecasterFitPredictMultivariateWithX,
    ForecasterFitPredictPanelSimple,
    ForecasterFitPredictHierarchicalSimple,
]

scenarios_forecasting = forecasting_scenarios_extended
