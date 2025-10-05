import pytest

from sktime.detection.base._base import BaseDetector
from sktime.registry import all_estimators
from sktime.utils._testing import scenarios_detectors


@pytest.mark.parametrize("Scenario", scenarios_detectors)
def test_detectors_run_scenarios(Scenario):
    """Run all detectors on simple defined scenarios."""
    scenario = Scenario()
    detectors = all_estimators(estimator_types="detector", return_names=False)

    if not detectors:
        pytest.skip("No detector estimators available.")

    for DetectorCls in detectors:
        detector = DetectorCls.create_test_instance()

        # Skip abstract or base detectors
        if not hasattr(detector, "_fit") or type(detector)._fit is BaseDetector._fit:
            continue

        fit_args = scenario.get_args("fit")

        try:
            detector.fit(**fit_args)
        except NotImplementedError:
            pytest.skip(f"{DetectorCls.__name__} is abstract or not fully implemented.")
