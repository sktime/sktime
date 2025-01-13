"""Signature moment transformer."""

import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel,delayed
from sktime.transformations.base import BaseTransformer

class SignatureMoments(BaseTransformer):
    """Signature Moments Transformer for multivariate time series.

    Computes time ordered signature moments for uni- and multivariate time series.

    Pipelining ``Differencer() * SignatureMoments()`` can be used to
    obtain the discrete path signature.

    For a degree ``d``, the columns of the ``transform`` output are
    strings corresponding to strings of length ``d`` over the alphabet
    ``{[0], ..., [str(n_channels - 1)]}``,
    where ``n_channels`` is the number of
    variables in the time series, and the character ``i`` represents the
    ``i``-th variable.

    If ``use_index=True``, the time index is included as an additional
    dimension, and is represented by the character ``[n_channels]``.

    For a time series :math:`X` with ``n_channels`` variables, the signature moment
    for a string :math:`s = i_1 i_2 ... i_d` is the arithmetic mean of the products

    .. math:: X_{i_1}(t_1) X_{i_2}(t_2) ... X_{i_d}(t_d)

    where :math:`t_1 < t_2 < ... < t_d` are the time indices.

    If ``normalize_prod=True``, the signature moment is computed as the arithmetic
    mean of the geometric means instead, i.e., of

    .. math:: (X_{i_1}(t_1) X_{i_2}(t_2) ... X_{i_d}(t_d))^{1/d}

    This ensures that all signature moments are of the same unit as the input data.

    Parameters
    ----------
    degree: int, default=2
        The maximum length of the string-based signature elements to include.
        Degree can be upto 3.

    use_index: bool, default=True
        Whether to include the time index as an additional dimension.

    normalize_prod: bool, default=False
        If True, uses geometric mean instead of product for the signature moment,
        see above for formula.
        If False, uses product.

    Examples
    --------
    >>> from sktime.transformations.series.signature import SignatureMoments
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SignatureMoments(degree=2, use_index=True)
    >>> Xt = transformer.fit_transform(y)
    """

    _tags = {
        "authors": ["AdiTyaPal0710","VectorNd", "fkiraly"],
        "maintainers": ["VectorNd","AdiTyaPal0710"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "np.ndarray",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
    }

    def __init__(self, degree=2, use_index=False,use_smaller_equal=False):
        if degree < 1:
            raise ValueError("`degree` must be an integer >= 1.")
        self.degree = degree
        self.use_index = use_index
        self.use_smaller_equal = use_smaller_equal
        super().__init__()

    def _transform(self, X, y=None):
        """Transform the input time series into signature moments."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        n_timepoints, n_channels = X.shape

        if self.use_index:
            time_index = np.arange(n_timepoints).reshape(-1, 1)
            X = np.hstack([X, time_index])  # Add time index as a feature
            n_channels += 1

        features = {}
        for degree in range(1, self.degree + 1):
            combs = product(range(n_channels))
            for comb in combs:
                column_name = "".join(map(str, comb))
                features[column_name] = self._compute_signature(X, comb)

        return pd.DataFrame([features])

    def _compute_signature(self, data, indices):
        """Compute the mean product for a given combination of dimensions."""
        selected_data = data[:, indices]
        n, k = selected_data.shape  # n = number of timepoints, k = combination size
        product_sum = 0.0
        count = 0.0

        # Generate all valid time index combinations of length k
        time_combinations = product(range(n), repeat=k)
        for time_comb in time_combinations:
            prod = 1.0
            for idx, time_idx in enumerate(time_comb):
                prod *= selected_data[time_idx, idx]
            product_sum += prod
            count += 1.0

        return product_sum / count if count > 0 else 0.0

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the transformer."""
        params0 = {"degree": 1, "use_index": False, "use_smaller_equal": False}
        params1 = {"degree": 2, "use_index": True, "use_smaller_equal": True}
        params2 = {"degree": 3, "use_index": False, "use_smaller_equal": True}
        return [params0, params1, params2]