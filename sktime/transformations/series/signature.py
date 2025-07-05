"""Signature moment transformer."""

import numpy as np
import pandas as pd

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
        "authors": ["VectorNd", "fkiraly"],
        "maintainers": "VectorNd",
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "np.ndarray",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
    }

    def __init__(self, degree=2, use_index=True, normalize_prod=False):
        self.degree = degree
        self.use_index = use_index
        self.normalize_prod = normalize_prod
        super().__init__()

    def _transform(self, X, y=None):
        """Compute the signature features for the input time series."""
        n_timepoints, n_channels = X.shape

        if self.use_index:
            index = np.arange(n_timepoints).reshape(-1, 1)
            X = np.concatenate((X, index), axis=1)
            n_channels += 1

        signature_matrix = []
        instance_data = X.T
        signature_row, feature_names = self._compute_signature(instance_data)
        signature_matrix.append(signature_row)

        if self.normalize_prod:
            signature_matrix = np.power(signature_matrix, 1 / self.degree)

        return pd.DataFrame(signature_matrix, columns=feature_names)

    def _compute_signature(self, data):
        """Compute signature features for a single instance."""
        n_channels, _ = data.shape

        feature_names = []
        signature_row = []
        for length in range(1, self.degree + 1):
            for indices in np.ndindex((n_channels,) * length):
                element_mean = self._compute_mean_product(data, indices)
                signature_row.append(element_mean)

                feature_name = "".join(f"[{i}]" for i in indices)
                feature_names.append(feature_name)

        return signature_row, feature_names

    def _compute_mean_product(self, data, indices):
        """Compute mean product of the specified data dimensions."""
        length = len(indices)
        ix = indices

        if length == 1:
            return np.mean(data[ix[0], :])
        elif length == 2:
            correlation_data = data[ix[0], :, np.newaxis] * data[ix[1], np.newaxis, :]
            triu_indices = np.triu_indices(correlation_data.shape[1])
            return np.mean(correlation_data[triu_indices])
        elif length == 3:
            correlation_data = (
                data[ix[0], :, np.newaxis, np.newaxis]
                * data[ix[1], np.newaxis, :, np.newaxis]
                * data[ix[2], np.newaxis, np.newaxis, :]
            )
            depth = correlation_data.shape[0]

            total_sum, total_len = 0, 0

            for i in range(depth):
                filtered_matrix = correlation_data[i][i:]

                triu_indices = np.triu_indices(filtered_matrix.shape[0])

                upper_triangular = filtered_matrix[triu_indices]

                total_len += len(upper_triangular)
                total_sum += np.sum(upper_triangular)

            data_mean = total_sum / total_len

            return data_mean
        else:
            raise NotImplementedError("Degree higher than 3 is not implemented.")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {"degree": 1, "use_index": True}
        params1 = {"degree": 3, "use_index": False}
        params2 = {"degree": 2, "use_index": True, "normalize_prod": True}
        return [params0, params1, params2]
