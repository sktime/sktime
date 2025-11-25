"""Test to ensure VM classes count is within GitHub Actions matrix limit."""

import pytest


def test_vm_classes_within_matrix_limit():
    """Test that number of VM classes is within GitHub Actions matrix limit.
    
    GitHub Actions has a maximum matrix size of 256 jobs.
    This test ensures _get_all_vm_classes returns 256 or fewer classes.
    """
    from sktime.tests.test_switch import _get_all_vm_classes

    vm_classes = _get_all_vm_classes()
    num_vm_classes = len(vm_classes)

    assert num_vm_classes <= 256, (
        f"Number of VM classes ({num_vm_classes}) exceeds GitHub Actions "
        f"matrix limit of 256. Consider splitting the test-vm job."
    )

