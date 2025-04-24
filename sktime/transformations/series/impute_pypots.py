"""PyPOTS imputer for partially observed time series."""

__author__ = ["Spinachboul", "kartik-555"]
__all__ = ["PyPOTSImputer"]

import os
from typing import Optional

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.dependencies._dependencies import _check_soft_dependencies


class PyPOTSImputer(BaseTransformer):
    """Imputer for partially observed time series using PyPOTS models.

    Parameters
    ----------
    model : str, default="SAITS"
        The PyPOTS model to use for imputation. Available models: "SAITS", "BRITS",
        "MRNN", "GPVAE", "Transformer".
    model_params : dict, optional
        Additional parameters for the PyPOTS model.
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    patience : int, default=10
        Early stopping patience.
    random_state : int, optional
        Random seed.

    Example
    -------
    >>> from sktime.transformations.series.impute_pypots import PyPOTSImputer
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import temporal_train_test_split
    >>> import numpy np
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> transformer = PyPOTSImputer(model="SAITS")
    >>> transformer.fit(y_train)
    PyPOTSImputer(...)
    >>> y_test.iloc[3] = np.nan
    >>> y_hat = transformer.transform(y_test)
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "handles-missing-data": True,
        "X_inner_mtype": ["pd.DataFrame", "pd.Series", "np.ndarray"],
        "y_inner_mtype": "None",
        "univariate-only": False,
        "requires_y": False,
        "fit_is_empty": False,
        "python_dependencies": "pypots",
    }

    def __init__(
        self,
        model: str = "SAITS",  # default model
        model_params: Optional[dict] = None,
        n_epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        # random_state: Optional[int] = None,
    ):
        self.model = model
        self.model_params = model_params if model_params is not None else {}
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        # self.random_state = random_state
        super().__init__()
        self._is_fitted = False
        self._imputer = None

    def _check_dependencies(self):
        """Check if pypots is installed."""
        _check_soft_dependencies("pypots", severity="error")
        import pypots

        return pypots

    def _fit(self, X, y=None):
        """Fit the imputer to the training data."""
        # Check dependencies
        self._check_dependencies()

        # Convert input to required format
        X_array, X_mask = self._prepare_input(X)

        models_list = ["SAITS", "BRITS", "MRNN", "GPVAE", "Transformer"]
        if self.model not in models_list:
            raise ValueError(
                f"Unknown model: {self.model}. Available models are: {models_list}"
            )

        pypots = self._check_dependencies()

        # Dynamically get model class from PyPOTS
        model_cls = getattr(pypots.imputation, self.model)

        # Get the dimension/feature size from input data
        n_features = X_array.shape[2] if X_array.ndim == 3 else X_array.shape[1]

        # Set up model parameters
        params = {
            "n_features": n_features,
            "batch_size": self.batch_size,
            "epochs": self.n_epochs,
            "patience": self.patience,
        }

        if self.random_state is not None:
            params["random_state"] = self.random_state

        # Update with user-defined parameters
        params.update(self.model_params)

        # Initialize and train the model
        self._imputer = model_cls(**params)

        # Skip if inside readthedocs build
        if os.environ.get("READTHEDOCS") == "True":
            self._is_fitted
            return self

        # Fit the model
        train_data = {"X": X_array, "missing_mask": X_mask}
        self._imputer.fit(train_data)

        self._is_fitted = True
        return self

    def _transform(self, X, y=None):
        """Transform X by imputing missing values."""
        if not self._is_fitted:
            raise ValueError("PyPOTSImputer is not fitted yet. Call 'fit' first.")

        # Check dependencies
        self._check_dependencies()

        X_array, X_mask = self._prepare_input(X)

        test_data = {"X": X_array, "missing_mask": X_mask}
        imputation_results = self._imputer.impute(test_data)

        imputed_data = imputation_results["imputation"]

        return self._restore_output(X, imputed_data)

    def _prepare_input(self, X):
        """Prepare input data for PyPOTS models."""
        if isinstance(X, pd.Series):
            X_array = X.to_numpy().reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            X_array = X.to_numpy()
        elif isinstance(X, np.ndarray):
            X_array = X.copy()
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
        else:
            raise TypeError(
                "Unsupported input type. Use pd.Series, pd.DataFrame, or np.ndarray."
            )

        # Create missing mask (True where values are missing)
        X_mask = np.isnan(X_array)

        # PyPOTS expects shape [batch_size, seq_len, n_features]
        if X_array.ndim == 2:
            X_array = np.expand_dims(X_array, axis=0)
            X_mask = np.expand_dims(X_mask, axis=0)

        return X_array, X_mask

    def _restore_output(self, X, imputed_data):
        """Restore output to the original format."""
        if isinstance(X, pd.Series):
            return pd.Series(imputed_data.flatten(), index=X.index)
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(imputed_data[0], index=X.index, columns=X.columns)
        else:
            return imputed_data

    def get_fitted_params(self):
        """Get fitted parameters."""
        if not self._is_fitted:
            raise ValueError("PyPOTSImputer is not fitted yet. Call 'fit' first.")

        return {"imputer": self._imputer}
