"""Test parity between metrics in sktime and skpro."""

import pytest

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


def _get_all_classes_in_module(module):
    """Get all classes defined in a module.

    Parameters
    ----------
    module : module
        The module to inspect.

    Returns
    -------
    dict
        A dictionary mapping class names to class objects defined in the module.
    """
    classes = {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    }
    return classes


@pytest.mark.skipif(not _check_soft_dependencies("skpro"), reason="requires skpro")
def test_skpro_sktime_metrics_parity():
    """Check that metrics in sktime are 1:1 copies of those in skpro.

    This is a parity check preceding a refactor where metrics move to skpro.
    """
    module_a = SKTIME_PROBA_METRICS
    module_b = SKPRO_PROBA_METRICS

    classes_a = _get_all_classes_in_module(module_a)
    classes_b = _get_all_classes_in_module(module_b)

    shared_classes = classes_a.keys() & classes_b.keys()

    for name in sorted(shared_classes):
        definition_a = _get_class_definition(classes_a[name])
        definition_b = _get_class_definition(classes_b[name])

        msg = f"Class {name} differs between sktime and skpro"

        assert definition_a == definition_b, msg
