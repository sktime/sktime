# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TSFEL transformer to extract features by domain or specific feature names."""

__author__ = ["Faakhir30"]


import pandas as pd

from sktime.transformations.base import BaseTransformer


class TSFELTransformer(BaseTransformer):
    """TSFEL transformer to extract features by domain or specific feature names.

    This transformer uses `features` parameter to extract features in following ways:

    1. By domain: Pass a domain ('statistical', 'temporal', 'spectral', 'fractal')
    2. By feature names: Pass feature function names (e.g., 'abs_energy', 'auc')
    3. Mixed: Pass a list containing both domain strings and feature names

    For domain-based extraction, uses TSFEL's `time_series_features_extractor`.
    For individual features, calls the feature functions directly from
    `tsfel.feature_extraction.features`.

    See tsfel documentation for available options for features and parameters.
    https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html

    Parameters
    ----------
    features : str, list of str, or None, optional (default=None)
        Features to extract. Can be:

        - A domain string: 'statistical', 'temporal', 'spectral', 'fractal'
        - A list of feature function names: ['abs_energy', 'auc', 'autocorr']
        - A list mixing domains and features: ['statistical', 'abs_energy']
        - None: extract all features from all domains

    kwargs : dict, optional (default=None)
        Additional keyword arguments passed to tsfel's feature extractor or
        individual feature functions. Common parameters include:

        - fs : float, sampling frequency
        - window_size : int, size of windows for feature extraction
        - overlap : float, overlap between windows (0-1)
        - verbose : int, verbosity level (0 or 1)
        - Any feature-specific parameters (e.g., percentile for ecdf_percentile_count)

    Examples
    --------
    >>> from sktime.transformations.series.tsfel import TSFELTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> # Extract all statistical domain features
    >>> transformer = TSFELTransformer(
    ...     features="statistical", kwargs={"verbose": 0}
    ... )
    >>> features = transformer.fit_transform(y)  # doctest: +SKIP
    >>> # Access TSFEL output for feature
    >>> transformer['statistical'].iloc[0]  # doctest: +SKIP
    >>> # Extract feature with custom parameters
    >>> transformer = TSFELTransformer(
    ...     features=["ecdf_percentile_count"],
    ...     kwargs={"percentile": [0.6, 0.9, 1.0], "verbose": 0}
    ... )
    >>> features = transformer.fit_transform(y)  # doctest: +SKIP
    >>> transformer['ecdf_percentile_count'].iloc[0]  # doctest: +SKIP
    """

    _tags = {
        "authors": ["Faakhir30"],
        "python_dependencies": ["tsfel"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
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
        "capability:categorical_in_X": False,
        # testing configuration
        # ---------------------
        "tests:vm": True,
    }

    def __init__(
        self,
        features=None,
        kwargs=None,
    ):
        # Call super().__init__() first to check soft dependencies
        super().__init__()

        self.domain_strings = ["statistical", "temporal", "spectral", "fractal"]
        self.features = features
        self.kwargs = kwargs

        # Validate features after initialization
        self._validate_features()

    def _get_features_list(self):
        """Normalize features to a list format for internal use."""
        if isinstance(self.features, str) or self.features is None:
            return [self.features]
        return self.features

    def _validate_features(self):
        """Validate that all features exist and required parameters are provided.

        Raises
        ------
        ValueError
            If a feature doesn't exist in tsfel.feature_extraction.features or
            if a feature requires a parameter that wasn't provided in kwargs.
        """
        import inspect
        from inspect import Parameter

        import tsfel.feature_extraction.features as tsfel_features

        features_list = self._get_features_list()
        for feature in features_list:
            # Skip domain strings and None
            if feature is None or feature in self.domain_strings:
                continue

            # Check if feature exists
            if not hasattr(tsfel_features, feature):
                raise ValueError(
                    f"Feature '{feature}' not found in "
                    "tsfel.feature_extraction.features. "
                    "Please check the feature name."
                )

            # Check if feature requires parameters that aren't provided
            feature_func = getattr(tsfel_features, feature)
            sig = inspect.signature(feature_func)
            sig_params = sig.parameters
            kwargs = {} if self.kwargs is None else self.kwargs

            for param_name, param in sig_params.items():
                if param_name == "signal":
                    continue

                # If parameter is not in kwargs and has no default, it's required
                if param_name not in kwargs and param.default == Parameter.empty:
                    raise ValueError(
                        f"Feature '{feature}' requires parameter '{param_name}' "
                        f"(positional or keyword) but it was not provided. "
                        f"Please provide it in the transformer kwargs, "
                        f"e.g., {param_name}=value."
                    )

    def _extract_individual_feature(self, X, feature_name):
        """Extract a single feature by calling individual feature functions directly.

        Returns the raw result from TSFEL feature function, preserving its
        natural format (scalar, Series, DataFrame, array, etc.). No format
        conversion is performed.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Input data
        feature_name : str
            Name of the feature function to extract (e.g., 'abs_energy', 'auc')

        Returns
        -------
        result : any
            Raw result from TSFEL feature function in its natural format.
            Can be scalar, Series, DataFrame, array, etc.
        """
        import inspect
        from inspect import Parameter

        import tsfel.feature_extraction.features as tsfel_features

        feature_func = getattr(tsfel_features, feature_name)

        # Analyze function signature
        sig = inspect.signature(feature_func)
        sig_params = sig.parameters
        # Build kwargs for parameters other than signal
        feature_kwargs = {}
        for param_name, param in sig_params.items():
            if param_name == "signal":
                continue

            if self.kwargs is not None and param_name in self.kwargs:
                feature_kwargs[param_name] = self.kwargs[param_name]
            elif param.default != Parameter.empty:
                feature_kwargs[param_name] = param.default

        # Handle multivariate data
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                # Process each column separately
                results = []
                for col in X.columns:
                    col_result = feature_func(X[col], **feature_kwargs)
                    results.append(col_result)
                return results
            else:
                # Single column DataFrame: make Series
                X = X.iloc[:, 0]
        
        # Univariate case
        result = feature_func(X, **feature_kwargs)
        return result

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        X_transformed : pd.DataFrame
            DataFrame containing extracted features. To access raw TSFEL output
            for a feature, use `transformer['feature_name'].iloc[0]` or
            `transformer[domain_name].iloc[0]` if no feature name or domain name
            is provided, use `transformer['all'].iloc[0]` to get all features.
        """
        import tsfel
        from tsfel.feature_extraction.calc_features import (
            time_series_features_extractor,
        )

        # Store raw results in a dictionary for dict-like access
        feature_results = {}

        features_list = self._get_features_list()
        for feature in features_list:
            if feature is None or feature in self.domain_strings:
                if feature:
                    cfg_file = tsfel.get_features_by_domain(feature)
                else:
                    cfg_file = tsfel.get_features_by_domain()
                domain_kwargs = {} if self.kwargs is None else self.kwargs
                domain_df = time_series_features_extractor(
                    cfg_file,
                    X,
                    **domain_kwargs,
                )
                feature_key = feature if feature is not None else "all"
                feature_results[feature_key] = domain_df
            else:
                # Individual features
                feature_results[feature] = self._extract_individual_feature(X, feature)

        # Use regular DataFrame for _transform to pass mtype checks
        return pd.DataFrame([feature_results], index=[0])

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"features": "statistical", "kwargs": {"verbose": 0}}
        params2 = {"features": "abs_energy"}
        params3 = {
            "features": ["abs_energy", "auc"],
            "kwargs": {"fs": 100, "verbose": 0},
        }
        # mixed features and domains
        params4 = {
            "features": ["statistical", "abs_energy"],
            "kwargs": {"verbose": 0},
        }

        return [params1, params2, params3, params4]
