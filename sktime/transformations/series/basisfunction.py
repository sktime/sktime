"""Radial Basis Function (RBF) Transformer for Time Series Data."""

__author__ = ["phoeenniixx"]
from enum import Enum

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class RBFType(Enum):
    """
    Enumeration for the types of Radial Basis Functions (RBFs) that can be applied.

    Available RBF types:
    - GAUSSIAN: Uses a Gaussian function.
    - MULTIQUADRIC: Uses a multiquadric function.
    - INVERSE_MULTIQUADRIC: Uses an inverse multiquadric function.
    """

    GAUSSIAN = "gaussian"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"


class RBFTransformer(BaseTransformer):
    """
    A custom transformer that applies Radial Basis Functions (RBFs) to time series data.

    This transformer allows the user to apply different RBFs such as Gaussian,
    multiquadric, and inverse multiquadric to time series data. It can be used
    as a feature transformation technique to augment or modify the input data
    in a machine learning pipeline.

    RBFs are functions that depend on the distance between a given point and
    a center point. The transformation involves applying a mathematical
    function (such as Gaussian or multiquadric) to this distance. Each
    center has an influence over the surrounding data points, and the
    influence decreases as the distance from the center increases. The
    `gamma` parameter controls the "spread" of the RBF,
    influencing how far the influence of each center reaches.

    The Gaussian RBF is calculated as `exp(-gamma * (x - c)^2)`, where `x` is
    the input data point and `c` is the center. Multiquadric and inverse multiquadric
    functions are calculated with different equations, providing different
    shapes for the transformation.

    This implementation is inspired by the `RepeatingBasisFunction` transformer
    from the `scikit-lego` package:
    https://github.com/koaning/scikit-lego/blob/main/sklego/preprocessing/repeatingbasis.py
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

    def __init__(self, centers=None, gamma=1.0, rbf_type=RBFType.GAUSSIAN):
        self.centers = centers
        self.gamma = gamma
        self.rbf_type = rbf_type
        self._fitted_centers = None
        super().__init__()

    def _rbf(self, x, c):
        """
        Compute the selected RBF for input x and center c.

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

        if self.rbf_type == RBFType.GAUSSIAN:
            return np.exp(-self.gamma * squared_diff)
        elif self.rbf_type == RBFType.MULTIQUADRIC:
            return np.sqrt(1 + self.gamma * squared_diff)
        elif self.rbf_type == RBFType.INVERSE_MULTIQUADRIC:
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

        if isinstance(X, (pd.DataFrame, pd.Series)):
            columns = [f"RBF_{i}" for i in range(X_transform.shape[1])]
            X_transform = pd.DataFrame(X_transform, index=X.index, columns=columns)

        return X_transform

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
                "rbf_type": RBFType.GAUSSIAN,
            },
            {
                "centers": np.linspace(-5, 5, num=3),
                "gamma": 1.0,
                "rbf_type": RBFType.MULTIQUADRIC,
            },
            {"centers": None, "gamma": 0.1, "rbf_type": RBFType.INVERSE_MULTIQUADRIC},
        ]
