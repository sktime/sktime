from sktime.transformations.base import BaseTransformer
import numpy as np
import pandas as pd 

class SignatureMomentsTransformer(BaseTransformer):
    """Signature Moments Transformer for multivariate time series.

    Computes discrete approximations of signature features for multivariate 
    time series, based on moments of combinations of dimensions. This is not 
    exactly the full path signature but captures important statistical moments.

    Parameters
    ----------
    degree: int, default=2
        The maximum length of the string-based signature elements to include.
        Degree can be upto 3.
    use_index: bool, default=True
        Whether to include the time index as an additional dimension.

    Attributes
    ----------
    signature_features_: list of str
        The list of signature feature names, based on combinations of dimensions.

    Notes
    -----
    The features computed here are not the full path signature, which would 
    typically involve pipelining with first differences. Instead, we compute 
    moments based on combinations of data dimensions. This is a discrete 
    approximation of signature features.
    """

    _tags = {
        "authors": "VectorNd",
        "maintainers": "VectorNd",
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
    }

    def __init__(self, degree=2, use_index=True):
        self.degree = degree
        self.use_index = use_index
        self.signature_features_ = []
        super().__init__()

    def _fit(self, X, y=None):
        """No fitting required for this transformer."""
        return self

    def _transform(self, X, y=None):
        """Compute the signature features for the input time series."""
        n_instances, n_channels, n_timepoints = X.shape

        if self.use_index:
            index = np.arange(n_timepoints).reshape(1, -1)
            X = np.concatenate((X, index[np.newaxis, :, :]), axis=1)
            n_channels += 1 

        signature_matrix = []
        for instance_idx in range(n_instances):
            instance_data = X[instance_idx]
            signature_row = self._compute_signature(instance_data)
            signature_matrix.append(signature_row)

        return pd.DataFrame(signature_matrix, columns = self.signature_features_)

    def _compute_signature(self, data):
        """Compute signature features for a single instance."""
        n_channels, n_timepoints = data.shape

        signature_row = []
        for length in range(1, self.degree + 1):
            for indices in np.ndindex((n_channels,) * length):
                element_mean = self._compute_mean_product(data, indices)
                signature_row.append(element_mean)

                feature_name = "".join(str(i + 1) for i in indices)
                self.signature_features_.append(feature_name)

        return signature_row

    def _compute_mean_product(self, data, indices):
        """Compute mean product of the specified data dimensions."""
        length = len(indices)
        n_timepoints = data.shape[1]

        if length == 1:
            return np.mean(data[indices[0], :])
        elif length == 2:
            correlation_data = data[indices[0], :, np.newaxis] * data[indices[1], np.newaxis, :]
            triu_indices = np.triu_indices(correlation_data.shape[1])
            return np.mean(correlation_data[triu_indices])
        elif length == 3:
            correlation_data = (data[indices[0], :, np.newaxis, np.newaxis] * 
                           data[indices[1], np.newaxis, :, np.newaxis] *
                           data[indices[2], np.newaxis, np.newaxis, :])
            depth = correlation_data.shape[0]

            total_sum , total_len = 0 , 0 

            for i in range(depth):
                filtered_matrix = correlation_data[i][i:]  

                triu_indices = np.triu_indices(filtered_matrix.shape[0])

                upper_triangular = filtered_matrix[triu_indices]

                total_len += len(upper_triangular)
                total_sum += np.sum(upper_triangular)

            data_mean = total_sum/total_len

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
        params = {
            "degree": 2,
            "use_index": True,
        }
        return params