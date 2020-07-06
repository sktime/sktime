"""
_checks.py
====================
Contains a reusable decorator function to handle the sklearn signature checks.
"""
import torch
import functools
from sktime.utils.validation.series_as_features import check_X_y, check_X
from sktime.utils.data_container import nested_to_3d_numpy


def handle_sktime_signatures(check_fitted=False):
    """Simple function for handling the sktime checks in signature modules.

    This decorator assumes that the input arguments to the function are either
    of the form:
        (self, data, labels)
    or
        (self, data).

    If this is in sktime format data, it will check the data and labels are of
    the correct form, and then
    """
    def real_decorator(func):
        """Reusable decorator to handle the sktime checks and convert the data
        to a torch Tensor.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Data checks
            args = list(args)
            if len(args) == 2:
                check_X(args[1], enforce_univariate=False)
            elif len(args) == 3:
                check_X_y(*args[1:], enforce_univariate=False)
            else:
                raise NotImplementedError(
                    "This decorator can only be used for args "
                    "(self, data, labels) or (self, data)."
                )
            # Fit checks
            if check_fitted:
                args[0].check_is_fitted()
            # Make tensor data
            args[1] = sktime_to_tensor(args[1])
            # Allow the function to be called on the checked and converted data
            return func(*args, **kwargs)
        return wrapper
    return real_decorator


def sktime_to_tensor(data):
    """Signature functionality requires torch tensors. This converts from
    the sktime format.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.Tensor(nested_to_3d_numpy(data)).transpose(1, 2)
    return data
