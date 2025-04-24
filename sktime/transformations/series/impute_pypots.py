"""PyPOTS imputer for partially observed time series."""

__author__ = ["Spinachboul", "kartik-555"]
__all__ = ["PyPOTSImputer"]

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.dependencies._dependencies import _check_soft_dependencies
from sktime.utils.validation import check_random_state


class PyPOTSImputer(BaseTransformer):
    """Imputer for partially observed time series using PyPOTS models.

    Parameters
    ----------
    model : str, default="SAITS"
        The PyPOTS model to use for imputation. Available models: "SAITS", "BRITS".
    model_params : dict, optional
        Additional parameters for the PyPOTS model.
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    patience : int, default=10
        Early stopping patience.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    Attributes
    ----------
    _supported_models : list[str]
        List of supported PyPOTS models.
    _default_model_params : dict[str, dict[str, Any]]
        Default parameters for each supported model.
    _required_model_params : dict[str, list[str]]
        Required parameters for each supported model.

    Example
    -------
    >>> from sktime.transformations.series.impute_pypots import PyPOTSImputer
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import temporal_train_test_split
    >>> import numpy as np
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

    # Define supported models
    _supported_models = ["SAITS", "BRITS"]

    # Define default parameters for each model
    _default_model_params = {
        "SAITS": {
            "n_layers": int,
            "n_steps": int,
            "n_features": int,
            "d_model": int,
            "n_heads": int,
            "d_k": int,
            "d_v": int,
            "d_ffn": int,
            "dropout": float,
            "attn_dropout": float,
            "diagonal_attention_mask": bool,
            "ORT_weight": float,
            "MIT_weight": float,
            "training_loss": str,
            "validation_metric": str,
        },
        "BRITS": {
            "n_steps": int,
            "n_features": int,
            "rnn_hidden_size": int,
            "training_loss": str,
            "validation_metric": str,
        },
    }

    # Required parameters for each model (excluding n_features which is auto-detected)
    _required_model_params = {
        "SAITS": [],
        "BRITS": [],
    }

    def __init__(
        self,
        model: str = "SAITS",
        model_params: Optional[dict] = None,
        n_epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        random_state: Optional[int] = None,
    ):
        self.model = model
        self.model_params = model_params if model_params is not None else {}
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        super().__init__()
        self._is_fitted = False
        self._imputer = None

    def _check_dependencies(self):
        """Check if pypots is installed."""
        _check_soft_dependencies("pypots", severity="error")
        import pypots

        return pypots

    def _validate_model(self):
        """Validate the selected model."""
        if self.model not in self._supported_models:
            raise ValueError(
                f"Unknown model: {self.model}."
                f"Available models are: {self._supported_models}"
            )

    def _prepare_model_params(self, n_features: int) -> dict[str, Any]:
        """Prepare model parameters with defaults and validation.

        Parameters
        ----------
        n_features : int
            Number of features in the data.

        Returns
        -------
        dict[str, Any]
            The prepared model parameters.
        """
        # Start with common parameters
        params = {
            "n_features": n_features,
            "batch_size": self.batch_size,
            "epochs": self.n_epochs,
            "patience": self.patience,
        }

        # Add default model-specific parameters
        model_defaults = self._default_model_params.get(self.model, {})
        params.update(model_defaults)

        # Add user-defined parameters (override defaults)
        params.update(self.model_params)

        # Handle random state
        if self.random_state is not None:
            params["random_state"] = check_random_state(self.random_state)

        # Check required parameters
        required_params = self._required_model_params.get(self.model, [])
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {self.model}: {missing_params}"
            )

        return params

    def _fit(self, X, y=None):
        """Fit the imputer to the training data."""
        # Check dependencies
        pypots = self._check_dependencies()

        # Validate model
        self._validate_model()

        # Convert input to required format
        X_array, X_mask = self._prepare_input(X)

        # Get the dimension/feature size from input data
        n_features = X_array.shape[2] if X_array.ndim == 3 else X_array.shape[1]

        # Prepare model parameters
        params = self._prepare_model_params(n_features)

        # Dynamically get model class from PyPOTS
        model_cls = getattr(pypots.imputation, self.model)

        # Initialize the model
        self._imputer = model_cls(**params)

        # Skip if inside readthedocs build
        if os.environ.get("READTHEDOCS") == "True":
            self._is_fitted = True
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
        """Prepare input data for PyPOTS models.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or np.ndarray
            Input data to prepare.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The prepared data array and missing mask.
        """
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
        """Restore output to the original format.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or np.ndarray
            Original input data.
        imputed_data : np.ndarray
            Imputed data from PyPOTS model.

        Returns
        -------
        pd.Series, pd.DataFrame, or np.ndarray
            Imputed data in the same format as X.
        """
        if isinstance(X, pd.Series):
            return pd.Series(imputed_data.squeeze(), index=X.index)
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(imputed_data[0], index=X.index, columns=X.columns)
        else:
            # Return in the same shape as input
            if X.ndim == 1:
                return imputed_data.squeeze()
            return imputed_data

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary of fitted parameters.
        """
        if not self._is_fitted:
            raise ValueError("PyPOTSImputer is not fitted yet. Call 'fit' first.")

        return {"imputer": self._imputer}
