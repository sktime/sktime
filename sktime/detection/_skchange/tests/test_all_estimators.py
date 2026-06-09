"""Tests for all skchange estimators, both detectors and interval scorers."""

from inspect import _empty, signature

import pytest
from skbase.base import BaseObject

from sktime.base import BaseEstimator
from sktime.detection._skchange.anomaly_detectors import ANOMALY_DETECTORS
from sktime.detection._skchange.anomaly_scores import ANOMALY_SCORES
from sktime.detection._skchange.change_detectors import CHANGE_DETECTORS
from sktime.detection._skchange.change_scores import CHANGE_SCORES
from sktime.detection._skchange.compose.penalised_score import PenalisedScore
from sktime.detection._skchange.costs import COSTS
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

DETECTORS = ANOMALY_DETECTORS + CHANGE_DETECTORS
INTERVAL_EVALUATORS = COSTS + CHANGE_SCORES + ANOMALY_SCORES + [PenalisedScore]
ESTIMATORS = DETECTORS + INTERVAL_EVALUATORS


@parametrize_with_checks(ESTIMATORS)
def test_sktime_compatible_estimators(obj, test_name):
    check_estimator(
        obj,
        tests_to_run=test_name,
        raise_exceptions=True,
        # The excluded tests fail in skchange due to the use of some custom tags
        # that are not in the VALID_ESTIMATOR_TAGS in sktime.
        #
        # test_estimator_tags is adjusted an implemented in this file.
        # test_valid_estimator_tags and test_valid_estimator_class_tags is implemented
        # in test_all_detectors.py and test_all_interval_scorers.py.
        tests_to_exclude=[
            "test_estimator_tags",
            "test_valid_estimator_tags",
            "test_valid_estimator_class_tags",
            "test_fit_does_not_overwrite_hyper_params",
            "test_non_state_changing_method_contract",
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    )


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_estimator_tags(estimator_class: type[BaseEstimator]):
    """Check conventions on estimator tags.

    Adapted from sktime.test_all_estimators.TestAllObjects.test_estimator_tags.
    """
    Estimator = estimator_class

    assert hasattr(Estimator, "get_class_tags")
    all_tags = Estimator.get_class_tags()
    assert isinstance(all_tags, dict)
    assert all(isinstance(key, str) for key in all_tags.keys())
    if hasattr(Estimator, "_tags"):
        tags = Estimator._tags
        msg = (
            f"_tags attribute of {estimator_class} must be dict, but found {type(tags)}"
        )
        assert isinstance(tags, dict), msg
        assert len(tags) > 0, f"_tags dict of class {estimator_class} is empty"

    # Avoid ambiguous class attributes
    ambiguous_attrs = ("tags", "tags_")
    for attr in ambiguous_attrs:
        assert not hasattr(Estimator, attr), (
            f"Please avoid using the {attr} attribute to disambiguate it from "
            f"estimator tags."
        )


@pytest.mark.parametrize("Estimator", ESTIMATORS)
def test_detector_no_mutable_defaults(Estimator: BaseEstimator):
    """Ensure no detectors have mutable default arguments."""

    detector = Estimator.create_test_instance()
    sig = signature(detector.__init__)
    mutable_types = (
        list,
        dict,
        set,
        BaseEstimator,
        BaseObject,
    )
    for param in sig.parameters.values():
        if param.default is not _empty and isinstance(param.default, mutable_types):
            raise AssertionError(
                f"Mutable default argument found in {Estimator.__name__}: {param.name}"
            )
