# -*- coding: utf-8 -*-
"""
_checks.py
====================
Contains a reusable decorator function to handle the sklearn signature checks.
"""
import functools
import numpy as np
import pandas as pd
from sktime.utils.validation.panel import check_X, check_X_y
from sktime.utils.data_processing import from_nested_to_3d_numpy


def _handle_sktime_signatures(check_fitted=False, force_numpy=False):
    """Simple function for handling the sktime checks in signature modules.

    This decorator assumes that the input arguments to the function are either
    of the form:
        (self, data, labels)
    or
        (self, data).

    Signature classes require numpy format data with dimensions to be of the
    form [batch, length channels]. This function performs the sktime checks,
    converts to numpy, and then converts back to the original format of the
    data.

    Args:
        check_fitted (bool): Set this to True to invoke sktimes `check_fitted`
            function. (For example when in a transform method).
        force_numpy (bool): Set True to force the output to be numpy. This is
            needed in prediction steps where we wish to output y as a numpy
            array.
    """

    def real_decorator(func):
        """Reusable decorator to handle the sktime checks and convert the data
        to numpy.
        """

        @functools.wraps(func)
        def wrapper(self, data, labels=None, **kwargs):
            # Check if pandas so we can convert back
            is_pandas = True if isinstance(data, pd.DataFrame) else False
            pd_idx = data.index if is_pandas else None

            # Fit checks
            if check_fitted:
                self.check_is_fitted()

            # First convert to pandas so everything is the same format
            if labels is None:
                data = check_X(data, coerce_to_pandas=True)
            else:
                data, labels = check_X_y(data, labels, coerce_to_pandas=True)

            # Now convert it to a numpy array
            # Note sktime uses [N, C, L] whereas signature code uses shape
            # [N, L, C] (C being channels) so we must transpose.
            data = np.transpose(from_nested_to_3d_numpy(data), [0, 2, 1])

            # Apply the function to the transposed array
            if labels is None:
                output = func(self, data, **kwargs)
            else:
                output = func(self, data, labels, **kwargs)

            # Convert back
            if all([is_pandas, isinstance(output, np.ndarray), not force_numpy]):
                output = pd.DataFrame(index=pd_idx, data=output)

            return output

        return wrapper

    return real_decorator
