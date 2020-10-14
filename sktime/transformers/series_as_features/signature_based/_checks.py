# -*- coding: utf-8 -*-
"""
_checks.py
====================
Contains a reusable decorator function to handle the sklearn signature checks.
"""
import functools
import pandas as pd
from sktime.utils.validation.series_as_features import check_X, check_X_y
from sktime.utils.data_container import from_nested_to_3d_numpy

from sktime.utils.check_imports import _check_soft_dependencies
_check_soft_dependencies('torch', 'signatory')  # noqa
import torch  # noqa: E402


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
                data = check_X(data, enforce_univariate=False, coerce_to_pandas=True)
            else:
                data, labels = check_X_y(
                    data, labels, enforce_univariate=False, coerce_to_pandas=True
                )
            # Make it a tensor, swap to [N, C, L] as this is sktime format
            # signature code assumes the channels are the end dimension
            data_idx = data.index
            if not isinstance(data, torch.Tensor):
                tensor_data = torch.Tensor(from_nested_to_3d_numpy(data)).transpose(
                    1, 2
                )
            else:
                tensor_data = data
            # Fit checks
            if check_fitted:
                self.check_is_fitted()
            # Allow the function to be called on the checked and converted data
            if labels is None:
                output = func(self, tensor_data, **kwargs)
            else:
                output = func(self, tensor_data, labels, **kwargs)
            if isinstance(output, torch.Tensor):
                output = pd.DataFrame(index=data_idx, data=output)
            return output

        return wrapper

    return real_decorator
