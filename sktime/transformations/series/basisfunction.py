"""Radial Basis Function (RBF) Transformer for Time Series Data."""

__author__ = ["phoeenniixx"]
import warnings

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class RBFTransformer(BaseTransformer):
    r"""A custom transformer to apply Radial Basis Functions (RBFs) to time series data.

    This transformer allows the user to apply various RBF kernels such as Gaussian,
    multiquadric, and inverse multiquadric to time series data. The transformation
    generates new features that augment the input data based on the distances between
    time points and specified center points, offering a flexible and non-linear feature
    representation for machine learning models.

    This implementation is inspired by the `RepeatingBasisFunction` transformer
    from the `scikit-lego` package:
    https://github.com/koaning/scikit-lego/blob/main/sklego/preprocessing/repeatingbasis.py

    Mathematical Background:
    Consider a time series represented as:
    - :math:`t_1, t_2, \dots, t_N`

    Each time point :math:`t_i` can be transformed into a new feature space via RBFs.
    The transformation computes the distance between the time points and a set of
    predefined "center points" :math:`c_1, \dots, c_K`. For each time point
    :math:`t_i`, the RBF is computed between :math:`t_i` and every center point
    :math:`c_k`, producing a matrix of transformed values. Each kernel function
    depends on the distance between a time point and a center point.

    Mathematically, the transformation for the Gaussian RBF is defined as:

        :math:`\phi(t_i, c_k) = \exp(-\gamma (t_i - c_k)^2)`

    where :math:`\gamma` is a scaling factor controlling the spread of the RBF.

    Similarly, other types of RBFs are available:
    - Multiquadric:
        :math:`\phi(t_i, c_k) = \sqrt{1 + \gamma (t_i - c_k)^2}`
    - Inverse Multiquadric:
        :math:`\phi(t_i, c_k) = \frac{1}{\sqrt{1 + \gamma (t_i - c_k)^2}}`

    These transformations produce new features for each time point, enhancing the
    representational power of the data by adding non-linear transformations that
    account for proximity to center points.

    Parameters
    ----------
    centers : array-like, shape (n_centers,), optional (default=None)
        The centers :math:`c_k` of the RBFs. These define the points against which
        the distances from the input data are measured. If `None`, the centers
        will be evenly spaced over the range of the input data.

    gamma : float, optional (default=1.0)
        The spread or scaling factor :math:`\gamma` that controls the influence
        range of each RBF center. Larger values of :math:`\gamma` make the RBF
        sharper (smaller spread), while smaller values make the RBF smoother.

    rbf_type : {"gaussian", "multiquadric", "inverse_multiquadric"},
                optional (default="gaussian")
        The type of radial basis function to apply:
        - "gaussian": :math:`\exp(-\gamma (t - c)^2)`
        - "multiquadric": :math:`\sqrt{1 + \gamma (t - c)^2}`
        - "inverse_multiquadric": :math:`\frac{1}{\sqrt{1 + \gamma (t - c)^2}}`

    apply_to : {"index", "values"}, optional (default="index")
        Determines whether the RBFs are applied to the time index or to the values
        of the time series.
        - "index": Apply the RBFs to the time index.
        - "values": Apply the RBFs to the values of the time series.

    use_torch : bool, optional (default=False)
    If True, the transformer will use PyTorch for the RBF calculations. If False,
    it will use NumPy. PyTorch needs to be installed if `use_torch=True`.

    Features Added
    --------------
    This transformer adds new features to the input time series. For each time point
    :math:`t_i` and each center point :math:`c_k`, a transformed value
    :math:`\phi(t_i, c_k)` is computed based on the chosen RBF.

    This produces a transformed dataset where each original time point is replaced with
    multiple transformed features, one for each center point.
    The number of new features generated for each time point equals the no. of centers.

    For example, if the original data has :math:`N` time points and :math:`K` centers,
    the transformed data will have :math:`N` rows and :math:`K` new features (columns).

    Attributes
    ----------
    _fitted_centers : array-like, shape (n_centers_,)
        The centers that are used for the RBF transformation. These are either provided
        by the user or computed from the data during fitting.

    _torch_available : bool
        Indicates if PyTorch is available. This is checked during fit.
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "X_inner_mtype": ["pd.DataFrame", "pd.Series", "np.ndarray"],
        "y_inner_mtype": "None",
        "univariate-only": False,
        "requires_y": False,
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "handles-missing-data": False,
        "python_dependencies": "torch",
        "authors": ["phoeenniixx"],
    }

    def __init__(
        self,
        centers=None,
        gamma=1.0,
        rbf_type="gaussian",
        apply_to="index",
        use_torch=False,
    ):
        self.centers = centers
        self.gamma = gamma
        self.rbf_type = rbf_type
        self.apply_to = apply_to
        self.use_torch = use_torch
        self._fitted_centers = None
        self._torch_available = None

        super().__init__()

    def _check_torch(self):
        """
        Check if PyTorch is available during fit.

        Must be called only in fit to maintain non-state-changing transform.
        """
        if not self.use_torch:
            self._torch_available = False
            return

        from importlib.util import find_spec

        self._torch_available = find_spec("torch") is not None

        if not self._torch_available:
            warnings.warn(
                "PyTorch is not available. Falling back to NumPy implementation. "
                "Install PyTorch to use the torch backend.",
                UserWarning,
            )

    def _get_torch(self):
        """
        Safely get the torch module when needed.

        Returns
        -------
        module
            The torch module if available and requested.

        Raises
        ------
        ImportError
            If torch is requested but not available.
        """
        if not self._torch_available:
            raise ImportError(
                "PyTorch operations requested but PyTorch is not available"
            )

        import torch

        return torch

    def _rbf_torch(self, x, c):
        """Compute the selected RBF using PyTorch."""
        torch = self._get_torch()

        x = torch.as_tensor(x, dtype=torch.float32)
        c = torch.as_tensor(c, dtype=torch.float32)

        x_reshaped = x.unsqueeze(-1)
        c_reshaped = c.reshape((1, 1, -1))

        squared_diff = (x_reshaped - c_reshaped) ** 2

        if self.rbf_type == "gaussian":
            result = torch.exp(-self.gamma * squared_diff)
        elif self.rbf_type == "multiquadric":
            result = torch.sqrt(1 + self.gamma * squared_diff)
        elif self.rbf_type == "inverse_multiquadric":
            result = 1 / torch.sqrt(1 + self.gamma * squared_diff)
        else:
            raise ValueError(f"Unsupported RBF type: {self.rbf_type}")

        return result.numpy()

    def _rbf_numpy(self, x, c):
        """Compute the selected RBF using NumPy."""
        x = np.atleast_2d(x)
        c = np.atleast_1d(c)

        x_reshaped = x[:, :, np.newaxis]
        c_reshaped = c[np.newaxis, np.newaxis, :]

        squared_diff = (x_reshaped - c_reshaped) ** 2

        if self.rbf_type == "gaussian":
            return np.exp(-self.gamma * squared_diff)
        elif self.rbf_type == "multiquadric":
            return np.sqrt(1 + self.gamma * squared_diff)
        elif self.rbf_type == "inverse_multiquadric":
            return 1 / np.sqrt(1 + self.gamma * squared_diff)
        else:
            raise ValueError(f"Unsupported RBF type: {self.rbf_type}")

    def _rbf(self, x, c):
        """
        Compute the selected RBF for input x and center c.

        Parameters
        ----------
        x : np.ndarray
            The input time series data, where rows correspond to time points and
            columns correspond to features.
        c : np.ndarray
            The centers for the RBFs.

        Returns
        -------
        np.ndarray
            The transformed data using the selected RBF kernel.
        """
        if self._torch_available:
            return self._rbf_torch(x, c)
        return self._rbf_numpy(x, c)

    def _fit(self, X, y=None):
        """
        Fit the RBF transformer by determining the centers for the RBFs.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or np.ndarray
            The input time series data.
        y : None
            Not used, present for API compatibility.

        Returns
        -------
        self
        """
        # Check torch availability during fit
        self._check_torch()

        if self.centers is None:
            if self.apply_to == "index":
                X_numeric = self._get_time_index(X)
            else:
                X_numeric = self._get_values(X)

            min_val = float(X_numeric.min())
            max_val = float(X_numeric.max())

            if self._torch_available:
                torch = self._get_torch()
                self._fitted_centers = torch.linspace(
                    min_val, max_val, steps=10
                ).numpy()
            else:
                self._fitted_centers = np.linspace(min_val, max_val, num=10)
        else:
            self._fitted_centers = np.array(self.centers)

        return self

    def _transform(self, X, y=None):
        """
        Transform the input time series data using the RBF transformation.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or np.ndarray
            The input time series data.
        y : None
            Not used, present for API compatibility.

        Returns
        -------
        X_transform : pd.DataFrame
            The transformed data, where each column corresponds to an RBF feature.
        """
        if self._fitted_centers is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")

        if self.apply_to == "index":
            input_data = self._get_time_index(X)
        else:
            input_data = self._get_values(X)

        X_transform = self._rbf(input_data, self._fitted_centers)

        n_samples, n_features, n_centers = X_transform.shape
        X_transform = X_transform.reshape(n_samples, n_features * n_centers)

        columns = [f"RBF_{i}_{j}" for i in range(n_features) for j in range(n_centers)]

        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.DataFrame(X_transform, index=X.index, columns=columns)
        return pd.DataFrame(X_transform, columns=columns)

    def _get_time_index(self, X):
        """
        Extract the time index from X and convert it to numeric values.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or np.ndarray
            The input time series data.

        Returns
        -------
        np.ndarray
            The time index values as a 2D array.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.to_numeric(X.index).values.reshape(-1, 1)
        elif isinstance(X, np.ndarray):
            return np.arange(X.shape[0]).reshape(-1, 1)
        else:
            raise ValueError(
                f"Unsupported input type: {type(X)}. "
                "Input X must be a pandas DataFrame, Series, or numpy array."
            )

    def _get_values(self, X):
        """
        Extract the values from X.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or np.ndarray
            The input time series data.

        Returns
        -------
        np.ndarray
            The values of the time series as a 2D array.
        """
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=[np.number]).values
        elif isinstance(X, np.ndarray):
            return X
        elif isinstance(X, pd.Series):
            return X.values.reshape(-1, 1)
        else:
            raise ValueError(
                f"Unsupported input type: {type(X)}. "
                "Input X must be a pandas DataFrame, Series, or numpy array."
            )

    @classmethod
    def get_test_params(cls):
        """Return test parameter sets for the transformer.

        Provide example parameters for unit testing or experimentation.

        Returns
        -------
        params : list of dict
            Each dictionary represents a set of params for initializing the transformer.
        """
        return [
            {
                "centers": np.linspace(0, 10, num=5),
                "gamma": 0.5,
                "rbf_type": "gaussian",
                "apply_to": "index",
                "use_torch": False,
            },
            {
                "centers": np.linspace(-5, 5, num=3),
                "gamma": 1.0,
                "rbf_type": "multiquadric",
                "apply_to": "values",
                "use_torch": True,
            },
            {
                "centers": None,
                "gamma": 0.1,
                "rbf_type": "inverse_multiquadric",
                "apply_to": "index",
                "use_torch": False,
            },
        ]
