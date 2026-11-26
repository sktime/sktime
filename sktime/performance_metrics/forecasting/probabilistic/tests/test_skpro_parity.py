"""Test parity between metrics in sktime and skpro."""

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

SKTIME_PROBA_METRICS = "sktime.performance_metrics.forecasting.probabilistic"
SKPRO_PROBA_METRICS = "skpro.metrics"


def _get_class_definition(cls):
    """Extract the relevant definition of a class."""
    import inspect

    members = {}

    for name, member in inspect.getmembers(cls):
        # Ignore automatically generated attributes
        if name.startswith("__") and name.endswith("__"):
            continue

        if inspect.isfunction(member) or inspect.ismethod(member):
            members[name] = inspect.getsource(member)
        elif not inspect.isroutine(member):
            members[name] = repr(member)

    return {
        "doc": inspect.getdoc(cls),
        "source": inspect.getsource(cls),
        "members": members,
    }


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="requires skpro",
)
def test_skpro_sktime_metrics_parity():
    """Check that metrics in sktime are 1:1 copies of those in skpro.

    This is a parity check preceding a refactor where metrics move to skpro.
    """
    from skpro.utils.retrieval import _all_classes

    module_a = SKTIME_PROBA_METRICS
    module_b = SKPRO_PROBA_METRICS

    classes_a = _all_classes(module_a)
    classes_b = _all_classes(module_b)

    classes_a_dict = {cls[0]: cls[1] for cls in classes_a}
    classes_b_dict = {cls[0]: cls[1] for cls in classes_b}

    shared_classes = classes_a_dict.keys() & classes_b_dict.keys()

    for name in sorted(shared_classes):
        definition_a = _get_class_definition(classes_a_dict[name])
        definition_b = _get_class_definition(classes_b_dict[name])

        msg = f"Class {name} differs between sktime and skpro"

        assert definition_a == definition_b, msg
