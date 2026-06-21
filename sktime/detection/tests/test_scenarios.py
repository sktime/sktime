"""Scenario definitions for detector tests."""

from sktime.utils._testing.scenarios_getter import retrieve_scenarios


def test_detector_scenarios_defined():
    """At least one detector scenario should be available via the scenario getter."""
    scenarios = retrieve_scenarios("detector")

    assert len(scenarios) > 0
    assert all(scenario.get_tag("scitype") == "detector" for scenario in scenarios)


def test_detector_multivariate_scenario_defined():
    """A multivariate detector scenario should be available for capable detectors."""
    scenarios = retrieve_scenarios("detector", filter_tags={"X_univariate": False})

    assert len(scenarios) > 0
