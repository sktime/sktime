"""Tests for parameter-dependent tags of foundation model forecasters.

Regression tests for bug #10520: ``set_tags`` calls made in ``__init__``
before ``super().__init__()`` are silently discarded, because the base class
``__init__`` re-initializes the dynamic tag overrides. Parameter-dependent
tags must instead be set in the ``__dynamic_tags__`` method.

None of these tests require soft dependencies to be present, since
tags are set at construction, and construction does not require soft
dependencies to be present.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pytest

from sktime.forecasting.chronos import ChronosForecaster
from sktime.forecasting.chronos2 import Chronos2Forecaster
from sktime.forecasting.mantis import MantisForecaster
from sktime.forecasting.timemoe import TimeMoEForecaster
from sktime.forecasting.timesfm import TimesFMForecaster
from sktime.tests.test_switch import run_test_module_changed

CLASSES_AND_KWARGS = [
    (Chronos2Forecaster, {}),
    (ChronosForecaster, {"model_path": "amazon/chronos-t5-tiny"}),
    (MantisForecaster, {}),
    (TimeMoEForecaster, {"model_path": "Maple728/TimeMoE-50M"}),
    (TimesFMForecaster, {}),
]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting"),
    reason="run test only incrementally (if requested)",
)
@pytest.mark.parametrize("cls, kwargs", CLASSES_AND_KWARGS)
def test_ignore_deps_clears_dependency_tag(cls, kwargs):
    """ignore_deps=True should clear the python_dependencies tag, see #10520."""
    est = cls(ignore_deps=True, **kwargs)

    deps = est.get_tag("python_dependencies", raise_error=False)
    assert deps in ([], None), (
        f"{cls.__name__}(ignore_deps=True) should clear the "
        f"python_dependencies tag, but get_tag returns {deps!r}"
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting"),
    reason="run test only incrementally (if requested)",
)
@pytest.mark.parametrize("cls, kwargs", CLASSES_AND_KWARGS)
def test_ignore_deps_survives_clone(cls, kwargs):
    """Dynamic dependency tags should be re-derived on clone."""
    est = cls(ignore_deps=True, **kwargs)

    deps = est.clone().get_tag("python_dependencies", raise_error=False)
    assert deps in ([], None), (
        f"clone of {cls.__name__}(ignore_deps=True) should clear the "
        f"python_dependencies tag, but get_tag returns {deps!r}"
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting"),
    reason="run test only incrementally (if requested)",
)
@pytest.mark.parametrize("cls, kwargs", CLASSES_AND_KWARGS)
def test_default_dependency_tag_unchanged(cls, kwargs):
    """Without ignore_deps, the class default dependency tags should apply."""
    est = cls(**kwargs)

    assert est.get_tag("python_dependencies", raise_error=False) == cls.get_class_tag(
        "python_dependencies"
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting"),
    reason="run test only incrementally (if requested)",
)
def test_chronos_use_source_package_dependency_tag():
    """use_source_package=True should switch the dependency set, see #10520."""
    est = ChronosForecaster(
        model_path="amazon/chronos-t5-tiny", use_source_package=True
    )

    assert est.get_tag("python_dependencies") == ["chronos"]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting"),
    reason="run test only incrementally (if requested)",
)
def test_timesfm_ignore_deps_clears_env_tags():
    """TimesFM ignore_deps=True should also clear python_version and env_marker."""
    est = TimesFMForecaster(ignore_deps=True)

    assert est.get_tag("python_version", raise_error=False) is None
    assert est.get_tag("env_marker", raise_error=False) is None


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting"),
    reason="run test only incrementally (if requested)",
)
def test_timesfm_broadcasting_tags():
    """TimesFM broadcasting=True should apply the broadcasting tag set, see #10520."""
    est = TimesFMForecaster(broadcasting=True, ignore_deps=True)

    assert est.get_tag("y_inner_mtype") == "pd.DataFrame"
    assert est.get_tag("X_inner_mtype") == "pd.DataFrame"
    assert not est.get_tag("capability:global_forecasting")
