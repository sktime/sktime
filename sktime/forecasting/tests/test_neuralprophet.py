# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Compatibility checks for the NeuralProphet interface."""

from packaging.requirements import Requirement

from sktime.forecasting.neuralprophet import NeuralProphet


def test_neuralprophet_pandas_version_requirement():
    """NeuralProphet should reject pandas 3 until the upstream API is compatible."""
    requirements = NeuralProphet.get_class_tag("python_dependencies")
    pandas_requirements = [
        Requirement(requirement)
        for requirement in requirements
        if Requirement(requirement).name == "pandas"
    ]

    assert len(pandas_requirements) == 1
    pandas_specifier = pandas_requirements[0].specifier
    assert pandas_specifier.contains("2.2.3")
    assert not pandas_specifier.contains("3.0.0")
