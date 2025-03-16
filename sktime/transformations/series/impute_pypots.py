"""PyPOTS imputer for partially observed time series."""

__author__ = ["Spinachboul", "kartik-555"]
__all__ = ["PyPOTSImputer"]

from typing import Optional

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.dependencies._dependencies import _check_soft_dependencies

# Check for PyPOTS at import time with a warning
_check_soft_dependencies("pypots", severity="warning")


class PyPOTSImputer(BaseTransformer):
    """Imputer for partially observed time series using PyPOTS models.

    The PyPOTSImputer transforms input series by replacing missing values according 
    to an imputation strategy specified by ``model``.

    Parameters
    ----------
    model : str, default="SAITS"
        The PyPOTS model to use for imputation. Available models are:
        "SAITS", "BRITS", "MRNN", "GPVAE", "Transformer".
    model_params : dict, optional
        Additional parameters to pass to the PyPOTS model. Default is None.
    n_epochs : int, default=100
        Number of epochs to train the PyPOTS model.
    batch_size : int, default=32
        Batch size for training the PyPOTS model.
    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
    random_state : int, optional
        Seed for the random number generator. Default is None.

    Examples
    --------
    >>> from sktime.transformations.series.impute_pypots import PyPOTSImputer
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import temporal_train_test_split
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

    def _fit(self, X, y=None):
        """Fit the imputer to the training data."""
        # Check at runtime to raise proper error
        _check_soft_dependencies("pypots", severity="error")

        # Convert input to required format
        X_array, X_mask = self._prepare_input(X)

        # Initialize the selected PyPOTS model
        if self.model == "SAITS":
            from pypots.imputation import SAITS

            model_cls = SAITS
        elif self.model == "BRITS":
            from pypots.imputation import BRITS

            model_cls = BRITS
        elif self.model == "MRNN":
            from pypots.imputation import MRNN

            model_cls = MRNN
        elif self.model == "GPVAE":
            from pypots.imputation import GPVAE

            model_cls = GPVAE
        elif self.model == "Transformer":
            from pypots.imputation import Transformer

            model_cls = Transformer
        else:
            raise ValueError(
                f"Unknown model: {self.model}. Available models: "
                "SAITS, BRITS, MRNN, GPVAE, Transformer"
            )

        # Get the dimension/feature size from input data
        n_features = X_array.shape[2] if X_array.ndim == 3 else X_array.shape[1]

        # Set up common model parameters
        params = {
            "n_features": n_features,
            "batch_size": self.batch_size,
            "epochs": self.n_epochs,
            "patience": self.patience,
        }

        # Add random state if provided
        if self.random_state is not None:
            params["random_state"] = self.random_state

        # Update with user-provided parameters
        params.update(self.model_params)

        # Initialize model
        self._imputer = model_cls(**params)

        # Prepare training data dictionary as expected by PyPOTS
        train_data = {"X": X_array, "missing_mask": X_mask}

        # Train the model - PyPOTS models expect a dictionary with X and missing_mask
        self._imputer.fit(train_data)

        self._is_fitted = True
        return self

    def _transform(self, X, y=None):
        """Transform X by imputing missing values.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Time series data to impute.
        y : None
            Ignored.

        Returns
        -------
        X_imputed : pd.Series or pd.DataFrame
            Imputed time series, same type as input.
        """
        if not self._is_fitted:
            raise ValueError("PyPOTSImputer is not fitted yet. Call 'fit' first.")

        _check_soft_dependencies("pypots", severity="error")

        # Convert input to required format
        X_array, X_mask = self._prepare_input(X)

        # Prepare test data dictionary as expected by PyPOTS
        test_data = {"X": X_array, "missing_mask": X_mask}

        # Perform imputation - PyPOTS models return a dictionary with imputation key
        imputation_results = self._imputer.impute(test_data)

        # Extract imputed data from the results
        imputed_data = imputation_results["imputation"]

        # Convert back to original format
        X_imputed = self._restore_output(X, imputed_data)

        return X_imputed

    def _prepare_input(self, X):
        """Prepare input data for PyPOTS models.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Input data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (data_array, missing_mask)
        """
        if isinstance(X, pd.Series):
            X_array = X.to_numpy().reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            X_array = X.to_numpy()
        else:  # numpy array
            X_array = X.copy()
            # Ensure 2D array
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)

        # Create missing mask (True where values are missing)
        X_mask = np.isnan(X_array)

        # Prepare data in the format PyPOTS expects
        # PyPOTS models typically expect data in shape [batch_size, seq_len, n_features]
        # For a single time series, batch_size=1
        if X_array.ndim == 2:
            # [seq_len, n_features] -> [1, seq_len, n_features]
            X_array = np.expand_dims(X_array, axis=0)
            X_mask = np.expand_dims(X_mask, axis=0)

        return X_array, X_mask

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        dict
            Dictionary of fitted parameters.
        """
        if not self._is_fitted:
            raise ValueError("PyPOTSImputer is not fitted yet. Call 'fit' first.")

        return {"imputer": self._imputer}
