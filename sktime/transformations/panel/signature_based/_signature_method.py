from sktime.transformations.base import BaseTransformer
import numpy as np

class SignatureTransformer(BaseTransformer):
    """Signature Transformer for multivariate time series.

    Computes signature features for multivariate time series, which are 
    time-ordered generalizations of moments.

    Parameters
    ----------
    degree: int, default=2
        The maximum length of the string-based signature elements to include.
    use_index: bool, default=True
        Whether to include the time index as an additional dimension.

    Attributes
    ----------
    signature_features_: list of str
        The list of signature feature names, based on combinations of dimensions.
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

        return np.array(signature_matrix)

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
            return np.mean([
                data[indices[0], i] * data[indices[1], j]
                for i in range(n_timepoints) for j in range(i + 1, n_timepoints)
            ])
        elif length == 3:
            return np.mean([
                data[indices[0], i] * data[indices[1], j] * data[indices[2], k]
                for i in range(n_timepoints)
                for j in range(i + 1, n_timepoints)
                for k in range(j + 1, n_timepoints)
            ])
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
