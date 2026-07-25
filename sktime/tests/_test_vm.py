"""Auxiliary script to test an estinator in its own virtual machine."""

__all__ = ["run_test_vm"]

import os
import platform
import re

from skbase.utils.dependencies import _check_soft_dependencies


def run_test_vm(cls_name):
    """Test an estimator in its own virtual machine.

    Takes a string which is the name of a class in the sktime registry,
    and runs ``check_estimator`` on it in a separate virtual machine,
    with deps determined by the tag ``python_dependencies`` of the class.

    Does not run the test if python and operating system versions
    are incompatible with the estimator's dependencies,
    as checked via ``_check_estimator_deps``.

    Parameters
    ----------
    cls_name : str
        Name of the estimator class to test, e.g., "ExampleForecaster".

    Raises
    -------
    Exception
        if the ``check_estimator`` fails, or if the estimator is not found.
    """
    from skbase.utils.dependencies import _check_estimator_deps

    from sktime.registry import craft
    from sktime.utils import check_estimator

    cls = craft(cls_name)
    if not _check_estimator_deps(cls, severity="none"):
        print(
            f"Skipping estimator: {cls} due to incompatibility "
            "with python or OS version."
        )
        return

    if _check_soft_dependencies("torch", severity="none"):
        # disable mps for macos runners if torch is available
        if platform.system() == "Darwin":
            import torch

            torch.backends.mps.is_available = lambda: False

    if _check_soft_dependencies("hf-xet", severity="none"):
        # to allow hf-xet to download models on macos runners on version `latest`
        if platform.system() == "Darwin":
            os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "4"

    skips = cls.get_class_tag("tests:skip_by_name", None)
    check_estimator(cls, raise_exceptions=True, tests_to_exclude=skips)


def _get_estimator_specific_test_modules(cls_name):
    """Get the list of modules to run for a specific estimator.

    Parameters
    ----------
    cls_name : str
        Name of the estimator class to test, e.g., "ExampleForecaster".

    Returns
    -------
    modules_to_run : list of str or None
        List of module paths to run for the estimator, or None if no specific
        modules are defined.
    """
    from importlib.util import find_spec

    from sktime.registry import craft

    cls = craft(cls_name)

    modules = cls.get_class_tag("tests:specific", None)
    if modules is None:
        return None

    msg = f"{cls.__name__}.tests:specific must be a list of strings, found: {modules}"
    assert isinstance(modules, list), msg
    assert all(isinstance(module, str) for module in modules), msg
    if len(modules) == 0:
        return None

    module_pat = re.compile(r"^sktime(?:\.[a-z_][a-z0-9_]*)*$")
    bad_modules = [module for module in modules if not module_pat.fullmatch(module)]
    assert len(bad_modules) == 0, (
        f"{cls.__name__}.tests:specific contains invalid module paths: {bad_modules}"
    )
    missing_modules = [module for module in modules if find_spec(module) is None]
    assert len(missing_modules) == 0, (
        f"{cls.__name__}.tests:specific contains missing modules: {missing_modules}"
    )

    modules_to_run = modules.copy()
    if len(modules_to_run) == 0:
        return None

    return modules_to_run
