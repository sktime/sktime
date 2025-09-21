"""Auxiliary script to test an estinator in its own virtual machine."""

__all__ = ["run_test_vm"]

import os
import platform

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
    from sktime.registry import craft
    from sktime.utils import check_estimator
    from sktime.utils.dependencies import _check_estimator_deps

    if _check_soft_dependencies("torch", severity="none"):
        # disable mps for macos runners if torch is available
        if platform.system() == "Darwin":
            import torch

            torch.backends.mps.is_available = lambda: False

    if _check_soft_dependencies("hf-xet", severity="none"):
        # to allow hf-xet to download models on macos runners on version `latest`
        if platform.system() == "Darwin":
            os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "4"

    cls = craft(cls_name)
    if _check_estimator_deps(cls, severity="none"):
        skips = cls.get_class_tag("tests:skip_by_name", None)
        check_estimator(cls, raise_exceptions=True, tests_to_exclude=skips)
    else:
        print(
            f"Skipping estimator: {cls} due to incompatibility "
            "with python or OS version."
        )
