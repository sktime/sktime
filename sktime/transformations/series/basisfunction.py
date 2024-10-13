"""Radial Basis Function (RBF) Transformer for Time Series Data."""

__author__ = ["phoeenniixx"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class RBFTransformer(BaseTransformer):
    """A custom transformer to apply Radial Basis Functions (RBFs) to time series data.

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
    - t_1, t_2, ..., t_N

    Each time point `t_i` can be transformed into a new feature space via RBFs. The
    transformation computes the distance between the time points and a set of predefined
    "center points" `c_1, ..., c_K`. For each time point `t_i`, the RBF is computed
    between `t_i` and every center point `c_k`, producing a matrix of transformed value.
    Each kernel function depends on the distance between a time point & a center point.

    Mathematically, the transformation for the Gaussian RBF is defined as:

        φ(t_i, c_k) = exp(-y * (t_i - c_k)^2)

    where `y` is a scaling factor controlling the spread of the RBF.

    Similarly, other types of RBFs are available:
    - Multiquadric: φ(t_i, c_k) = sqrt(1 + y * (t_i - c_k)^2)
    - Inverse Multiquadric: φ(t_i, c_k) = 1 / sqrt(1 + y * (t_i - c_k)^2)

    These transformations produce new features for each time point, enhancing the
    representational power of the data by adding non-linear transformations that
    account for proximity to center points.

    Parameters
    ----------
    centers : array-like, shape (n_centers,), optional (default=None)
        The centers `c_k` of the RBFs. These define the points against which
        the distances from the input data are measured. If `None`, the centers
        will be evenly spaced over the range of the input data.

    gamma : float, optional (default=1.0)
        The spread or scaling factor that controls the influence range of each
        RBF center. Larger values of `gamma` make the RBF sharper (smaller spread),
        while smaller values make the RBF smoother (larger spread).

    rbf_type : {"gaussian", "multiquadric", "inverse_multiquadric"},
                optional (default="gaussian")
        The type of radial basis function to apply:
        - "gaussian": exp(-gamma * (t - c)^2)
        - "multiquadric": sqrt(1 + gamma * (t - c)^2)
        - "inverse_multiquadric": 1 / sqrt(1 + gamma * (t - c)^2).

    Features Added
    ----------------
    This transformer adds new features to the input time series. For each time point
    `t_i` and each center point `c_k`, a transformed value `φ(t_i, c_k)` is computed
    based on the chosen RBF. This produces a transformed dataset where each original
    time point is replaced with multiple transformed features, one for each center pt.

    The number of new features generated for each time point equals the no of centers.
    For example, if the original data has N time points and K centers, the transformed
    data will have N rows and K new features (columns).

    Attributes
    ----------
    _fitted_centers : array-like, shape (n_centers_,)
        The centers that are used for the RBF transformation. These are either provided
        by the user or computed from the data during fitting.
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
        "authors": ["phoeenniixx"],
    }

    def __init__(self, centers=None, gamma=1.0, rbf_type="gaussian"):
        self.centers = centers
        self.gamma = gamma
        self.rbf_type = rbf_type
        self._fitted_centers = None
        super().__init__()

    def _rbf(self, x, c):
        """Compute the selected RBF for input x and center c.

        Parameters
        ----------
        x : scalar, array-like, shape (n_samples,)
            The input data (time series or single value).
        c : array-like, shape (n_centers,)
            The centers of the RBF.

        Returns
        -------
        array-like
            The transformed data using the selected RBF kernel.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        c = np.asarray(c, dtype=np.float64)

        x = x.reshape(-1, 1)
        c = c.reshape(1, -1)

        squared_diff = (x - c) ** 2

        if self.rbf_type == "gaussian":
            return np.exp(-self.gamma * squared_diff)
        elif self.rbf_type == "multiquadric":
            return np.sqrt(1 + self.gamma * squared_diff)
        elif self.rbf_type == "inverse_multiquadric":
            return 1 / np.sqrt(1 + self.gamma * squared_diff)
        else:
            raise ValueError(f"Unsupported RBF type: {self.rbf_type}")

    def _fit(self, X, y=None):
        """Fit the RBF transformer.

        Parameters
        ----------
        X : pd.DataFrame, np.ndarray, or pd.Series
            Input time series data.

        Returns
        -------
        self
        """
        if self.centers is None:
            if isinstance(X, pd.DataFrame):
                X_numeric = X.select_dtypes(include=[np.number]).values.flatten()
            elif isinstance(X, np.ndarray):
                X_numeric = X.flatten()
            elif isinstance(X, pd.Series):
                X_numeric = X.values
            else:
                raise ValueError(
                    f"Unsupported input type: {type(X)}.\
                     Input X must be a pandas DataFrame, Series, or numpy array."
                )

            self._fitted_centers = np.linspace(X_numeric.min(), X_numeric.max(), num=10)
        else:
            self._fitted_centers = np.array(self.centers)

        return self

    def _transform(self, X, y=None):
        """Transform the input time series data using the RBF transformation.

        Apply the selected RBF to each time series or value in the input data.

        Parameters
        ----------
        X : pd.DataFrame, np.ndarray, or pd.Series
            Input time series data to be transformed.

        Returns
        -------
        X_transform : pd.DataFrame or np.ndarray
            The transformed time series data.
        """
        if self._fitted_centers is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")

        if isinstance(X, pd.DataFrame):
            X_transform = X.apply(
                lambda col: self._rbf(col.values, self._fitted_centers).flatten()
            )
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X_transform = self._rbf(X, self._fitted_centers)
            else:
                X_transform = np.apply_along_axis(
                    lambda x: self._rbf(x, self._fitted_centers).flatten(),
                    axis=1,
                    arr=X,
                )
        elif isinstance(X, pd.Series):
            X_transform = self._rbf(X.values, self._fitted_centers)
        else:
            raise ValueError(
                f"Unsupported input type: {type(X)}.\
                 Input X must be a pandas DataFrame, Series, or numpy array."
            )

        columns = [f"RBF_{i}" for i in range(X_transform.shape[1])]
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.DataFrame(X_transform, index=X.index, columns=columns)
        else:
            return pd.DataFrame(X_transform, columns=columns)

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
            },
            {
                "centers": np.linspace(-5, 5, num=3),
                "gamma": 1.0,
                "rbf_type": "multiquadric",
            },
            {"centers": None, "gamma": 0.1, "rbf_type": "inverse_multiquadric"},
        ]
