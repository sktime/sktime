# -*- coding: utf-8 -*-
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

from sktime.base import BaseObject
from sktime.forecasting.base import BaseForecaster
from sktime.utils._testing.forecasting import _make_series
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42

# number timepoints used for training in all scenarios
NUM_T = 10


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
        if not isinstance(obj, BaseForecaster) and not issubclass(obj, BaseForecaster):
            return False

        # applicable only if number of variables in y complies with scitype:y
        is_univariate = self.get_tag("univariate_y")

        if is_univariate and get_tag(obj, "scitype:y") == "multivariate":
            return False

        if not is_univariate and get_tag(obj, "scitype:y") == "univariate":
            return False

        # applicable only if fh is not passed later than it needs to be
        fh_in_fit = self.get_tag("fh_passed_in_fit")

        if not fh_in_fit and get_tag(obj, "requires-fh-in-fit"):
            return False

        # if scenario contains NA, applicable only if forecaster can handle the
        scenario_has_nans = self.get_tag("has_nans", False)

        if scenario_has_nans and not get_tag(obj, "handles-missing-data"):
            return False

        return True


class ForecasterFitPredictUnivariateNoX(ForecasterTestScenario):
    """Fit/predict only, univariate y, no X."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True, "pre-refactor": True}

    args = {
        "fit": {"y": _make_series(n_timepoints=NUM_T, random_state=RAND_SEED), "fh": 1},
        "predict": {"fh": 1},
    }
    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateNoXEarlyFh(ForecasterTestScenario):
    """Fit/predict only, univariate y, no X, no fh in predict."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True}

    args = {
        "fit": {"y": _make_series(n_timepoints=NUM_T, random_state=RAND_SEED), "fh": 1},
        "predict": {},
    }
    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateNoXLateFh(ForecasterTestScenario):
    """Fit/predict only, univariate y, no X, no fh in predict."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": False}

    args = {
        "fit": {"y": _make_series(n_timepoints=NUM_T, random_state=RAND_SEED)},
        "predict": {"fh": 1},
    }
    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateNoXLongFh(ForecasterTestScenario):
    """Fit/predict only, univariate y, no X, longer fh."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True}

    args = {
        "fit": {
            "y": _make_series(n_timepoints=NUM_T, random_state=RAND_SEED),
            "fh": [1, 2, 3],
        },
        "predict": {},
    }
    default_method_sequence = ["fit", "predict"]


LONG_X = _make_series(n_columns=2, n_timepoints=2 * NUM_T, random_state=RAND_SEED)
X = LONG_X.iloc[0:NUM_T]
X_test = LONG_X.iloc[NUM_T : (NUM_T + 3)]
X_test_short = LONG_X.iloc[NUM_T : (NUM_T + 1)]


class ForecasterFitPredictUnivariateWithX(ForecasterTestScenario):
    """Fit/predict only, univariate y, with X."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True}

    args = {
        "fit": {
            "y": _make_series(n_timepoints=NUM_T, random_state=RAND_SEED),
            "X": X.copy(),
            "fh": 1,
        },
        "predict": {"X": X_test_short.copy()},
    }
    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictUnivariateWithXLongFh(ForecasterTestScenario):
    """Fit/predict only, univariate y, with X, and longer fh."""

    _tags = {"univariate_y": True, "fh_passed_in_fit": True}

    args = {
        "fit": {
            "y": _make_series(n_timepoints=NUM_T, random_state=RAND_SEED),
            "X": X.copy(),
            "fh": [1, 2, 3],
        },
        "predict": {"X": X_test.copy()},
    }
    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictMultivariateNoX(ForecasterTestScenario):
    """Fit/predict only, multivariate y, no X."""

    _tags = {"univariate_y": False, "fh_passed_in_fit": True, "pre-refactor": True}

    args = {
        "fit": {
            "y": _make_series(n_timepoints=NUM_T, n_columns=2, random_state=RAND_SEED),
            "fh": 1,
        },
        "predict": {},
    }
    default_method_sequence = ["fit", "predict"]


class ForecasterFitPredictMultivariateWithX(ForecasterTestScenario):
    """Fit/predict only, multivariate y, with X, and longer fh."""

    _tags = {"univariate_y": False, "fh_passed_in_fit": True}

    args = {
        "fit": {
            "y": _make_series(n_timepoints=NUM_T, n_columns=2, random_state=RAND_SEED),
            "X": X.copy(),
            "fh": [1, 2, 3],
        },
        "predict": {"X": X_test.copy()},
    }
    default_method_sequence = ["fit", "predict"]


y_nan = _make_series(n_timepoints=NUM_T, n_columns=1, random_state=RAND_SEED)
y_nan.iloc[0] = None
y_nan.iloc[NUM_T - 1, 0] = None
X_nan = X.copy()
X_nan.iloc[0, 0] = None
X_nan.iloc[NUM_T - 1, 0] = None


class ForecasterFitPredictWithNan(ForecasterTestScenario):
    """Fit/predict only, univariate y, with X, fh passed early."""

    _tags = {
        "univariate_y": True,
        "has_nans": True,
        "fh_passed_in_fit": True,
        "pre-refactor": True,
    }

    args = {
        "fit": {"y": y_nan, "X": X_nan, "fh": [1, 2]},
        "predict": {"X": X_test.copy()},
    }
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
    ForecasterFitPredictWithNan,
]

scenarios_forecasting = forecasting_scenarios_extended
