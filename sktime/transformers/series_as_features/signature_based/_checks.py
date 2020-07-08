"""
_checks.py
====================
Contains a reusable decorator function to handle the sklearn signature checks.
"""
import functools
import torch
import pandas as pd
from sktime.utils.validation.series_as_features import check_X, check_X_y
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
        def wrapper(self, data, labels=None, **kwargs):
            # Data checks
            if labels is None:
                check_X(data, enforce_univariate=False)
            else:
                check_X_y(data, labels, enforce_univariate=False)
            # Fit checks
            if check_fitted:
                self.check_is_fitted()
            # Keep the dataframe index as output is required to be of the
            # same format
            data_idx = data.index
            # Make tensor data
            tensor_data = sktime_to_tensor(data)
            # Allow the function to be called on the checked and converted data
            if labels is None:
                output = func(self, tensor_data, **kwargs)
            else:
                output = func(self, tensor_data, labels, **kwargs)
            # Rebuild into a dataframe if the output is a tensor
            if isinstance(output, torch.Tensor):
                output = pd.DataFrame(index=data_idx, data=output)
            return output
        return wrapper
    return real_decorator


def sktime_to_tensor(data):
    """Signature functionality requires torch tensors. This converts from
    the sktime format.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.Tensor(nested_to_3d_numpy(data)).transpose(1, 2)
    return data
