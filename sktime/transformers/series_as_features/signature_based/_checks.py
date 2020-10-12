"""
_checks.py
====================
Contains a reusable decorator function to handle the sklearn signature checks.
"""
import functools
import torch
import pandas as pd
import numpy as np
from sktime.utils.validation.series_as_features import check_X, check_X_y
from sktime.utils.data_container import from_nested_to_3d_numpy


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
            is_df = isinstance(data, pd.DataFrame)
            is_arr = isinstance(data, (np.ndarray, np.generic))
            is_tens = isinstance(data, torch.Tensor)
            assert any([is_df, is_arr, is_tens]), (
                "Signature methods only "
                "accept sktime dataframe "
                "format, numpy arrays or "
                "pytorch Tensors."
            )
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
            if is_df:
                data_idx = data.index
                # Make tensor data
                tensor_data = sktime_to_tensor(data)
            if is_arr:
                tensor_data = torch.Tensor(data).transpose(1, 2)
            # Allow the function to be called on the checked and converted data
            if labels is None:
                output = func(self, tensor_data, **kwargs)
            else:
                output = func(self, tensor_data, labels, **kwargs)
            # Rebuild into a dataframe if the output is a tensor
            if all([is_df, isinstance(output, torch.Tensor)]):
                output = pd.DataFrame(index=data_idx, data=output)
            if all([is_arr, isinstance(output, torch.Tensor)]):
                output = output.numpy()
            return output

        return wrapper

    return real_decorator


def sktime_to_tensor(data):
    """Signature functionality requires torch tensors. This converts from
    the sktime format.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.Tensor(from_nested_to_3d_numpy(data)).transpose(1, 2)
    return data
