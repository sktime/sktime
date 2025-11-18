# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base adapter for TSFEL individual feature transformers."""

__author__ = ["Faakhir30"]

import numpy as np
import pandas as pd
from inspect import Parameter, signature

from sktime.transformations.base import BaseTransformer


class _TSFELFeatureAdapter(BaseTransformer):
    """Base adapter class for TSFEL individual feature functions.

    This adapter wraps TSFEL feature extraction functions to work as sktime transformers.
    TSFEL features can return scalars (float/int) or arrays (nd-array).

    Parameters
    ----------
    feature_func : callable
        The TSFEL feature function to wrap (e.g., tsfel.feature_extraction.features.abs_energy)
    feature_name : str
        Name of the feature (used for column naming in output)
    fs : float, optional (default=None)
        Sampling frequency. Only used if the feature function requires it.
    **kwargs : dict
        Additional keyword arguments to pass to the feature function.
        These override any default values in the feature function signature.

    Notes
    -----
    - For DataFrame input, features return a Series with one value per column
    - For Series input, features return a scalar or array
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Faakhir30"],
        "python_dependencies": ["tsfel"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # scitype:transform-output is set dynamically in __init__ dynamically
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        #
        # behavioural tags: internal type
        # ----------------------------------
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "capability:multivariate": True,
        "requires_y": False,
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "capability:missing_values": False,
    }

    def __init__(
        self, feature_func, feature_name, output_type="Primitives", fs=None, **kwargs
    ):
        self.feature_func = feature_func
        self.feature_name = feature_name
        self.fs = fs
        self.kwargs = kwargs

        super().__init__()

        sig = signature(feature_func)
        self._sig = sig

        # Check if feature function requires fs parameter
        self._requires_fs = "fs" in sig.parameters
        if self._requires_fs:
            fs_param = sig.parameters.get("fs")
            # Check if fs is positional (not keyword-only)
            self._fs_is_positional = fs_param is not None and fs_param.kind in (
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.POSITIONAL_ONLY,
            )
            # Check if fs has a default value
            self._fs_has_default = (
                fs_param is not None and fs_param.default != Parameter.empty
            )
        else:
            self._fs_is_positional = False
            self._fs_has_default = False

        # Store parameter defaults from function signature (excluding signal and fs)
        self._func_defaults = {}
        for param_name, param in sig.parameters.items():
            if param_name not in ("signal", "fs") and param.default != Parameter.empty:
                self._func_defaults[param_name] = param.default

        self.set_tags(**{"scitype:transform-output": output_type})

    def _prepare_feature_args(self, X):
        """Prepare arguments for calling the feature function.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or np.ndarray
            Input signal(s) - TSFEL handles DataFrame/Series natively

        Returns
        -------
        args : tuple
            Positional arguments for the feature function
        kwargs : dict
            Keyword arguments for the feature function
        """
        # Start with defaults from function signature
        func_kwargs = self._func_defaults.copy()

        # Override with user-provided kwargs
        func_kwargs.update(self.kwargs)

        # Handle fs parameter
        args = (X,)
        if self._requires_fs:
            if self.fs is not None:
                if self._fs_is_positional:
                    # fs is positional, add as second positional argument
                    args = (X, self.fs)
                else:
                    # fs is keyword-only, add to kwargs
                    func_kwargs["fs"] = self.fs
            elif not self._fs_has_default:
                # fs is required but not provided - this will raise an error when called
                # but we let the function handle it for clearer error messages
                pass

        return args, func_kwargs

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            Data to be transformed. Can be pd.Series, pd.DataFrame, or np.ndarray.
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        X_transformed : pd.Series, pd.DataFrame, scalar, or array
            Result from TSFEL feature function. Format depends on feature function
            and input type.
        """

        # Prepare arguments for feature function
        args, func_kwargs = self._prepare_feature_args(X)

        result = self.feature_func(*args, **func_kwargs)

        return result
