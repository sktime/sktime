# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TSFEL transformer to extract features by domain or specific feature names."""

__author__ = ["Faakhir30"]


from sktime.transformations.base import BaseTransformer
import pandas as pd


class _TSFELDataFrame(pd.DataFrame):
    """DataFrame to store TSFEL feature results."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        """Get the raw result for a feature."""
        return super().__getitem__(key)[0]


class TSFELTransformer(BaseTransformer):
    """TSFEL transformer to extract features by domain or specific feature names.

    This transformer can extract features in two ways:
    1. By domain: Pass a domain string ('statistical', 'temporal', 'spectral', 'fractal')
    2. By feature names: Pass feature function names (e.g., 'abs_energy', 'auc')
    3. Mixed: Pass a list containing both domain strings and feature names

    For domain-based extraction, uses TSFEL's `time_series_features_extractor`.
    For individual features, calls the feature functions directly from
    `tsfel.feature_extraction.features`.

    Parameters
    ----------
    features : str, list of str, or None, optional (default=None)
        Features to extract. Can be:
        - A domain string: 'statistical', 'temporal', 'spectral', 'fractal'
        - A list of feature function names: ['abs_energy', 'auc', 'autocorr']
        - A list mixing domains and features: ['statistical', 'abs_energy']
        - None: extract all features from all domains
    **kwargs : dict
        Additional keyword arguments passed to tsfel's feature extractor or
        individual feature functions. Common parameters include:
        - fs : float, sampling frequency
        - window_size : int, size of windows for feature extraction
        - overlap : float, overlap between windows (0-1)
        - verbose : int, verbosity level (0 or 1)
        - Any feature-specific parameters (e.g., percentile for ecdf_percentile_count)
        See tsfel documentation for available options.

    Examples
    --------
    >>> from sktime.transformations.series.tsfel import TSFELTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> # Extract all statistical domain features
    >>> transformer = TSFELTransformer(features="statistical")
    >>> features = transformer.fit_transform(y)
    >>> # Extract specific features
    >>> transformer = TSFELTransformer(
    ...     features=["abs_energy", "auc", "autocorr"],
    ...     fs=100,
    ...     window_size=100,
    ...     verbose=0
    ... )
    >>> features = transformer.fit_transform(y)
    >>> # Extract feature with custom parameters
    >>> transformer = TSFELTransformer(
    ...     features=["ecdf_percentile_count"],
    ...     percentile=[0.6, 0.9, 1.0]
    ... )
    >>> features = transformer.fit_transform(y)
    >>> # Mix domains and individual features
    >>> transformer = TSFELTransformer(features=["statistical", "abs_energy"])
    >>> features = transformer.fit_transform(y)
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
        # testing configuration
        # ---------------------
        "tests:vm": True,
    }

    def __init__(
        self,
        features=None,
        **kwargs,
    ):
        self.features = features
        if isinstance(self.features, str) or self.features is None:
            self.features = [self.features]
        self.kwargs = dict(kwargs)

        super().__init__()

    def _extract_individual_feature(self, X, feature_name):
        """Extract a single feature by calling individual feature functions directly.

        Returns the raw result from TSFEL feature function, preserving its natural format
        (scalar, Series, DataFrame, array, etc.). No format conversion is performed.

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
        import tsfel.feature_extraction.features as tsfel_features
        import inspect
        from inspect import Parameter
        import pandas as pd
        import numpy as np

        # Check if feature exists in tsfel.feature_extraction.features
        if not hasattr(tsfel_features, feature_name):
            raise ValueError(
                f"Feature '{feature_name}' not found in tsfel.feature_extraction.features. "
                "Please check the feature name."
            )

        feature_func = getattr(tsfel_features, feature_name)

        # Analyze function signature
        sig = inspect.signature(feature_func)
        sig_params = sig.parameters
        # Build kwargs for non-signal parameters
        feature_kwargs = {}
        for param_name, param in sig_params.items():
            if param_name == "signal":
                continue

            if param_name in self.kwargs:
                feature_kwargs[param_name] = self.kwargs[param_name]
            elif param.default != Parameter.empty:
                feature_kwargs[param_name] = param.default
            else:
                raise ValueError(
                    f"Feature '{feature_name}' requires parameter '{param_name}' "
                    f"(positional or keyword) but it was not provided. "
                    f"Please provide it in the transformer kwargs, e.g., {param_name}=value."
                )

        # Call feature function: signal as first positional, others as kwargs
        result = feature_func(X, **feature_kwargs)

        # Return raw result - no conversion
        return result

    def fit_transform(self, X, y=None):
        """Transform X and return a transformed version.

        Stores raw TSFEL results internally. Access raw results via the transformer's
        `get_feature_result()` method to get each feature's natural output format.

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
            for a feature, use `transformer.get_feature_result('feature_name')`.
        """
        import tsfel
        import pandas as pd
        import numpy as np

        domain_strings = ["statistical", "temporal", "spectral", "fractal"]

        # Store raw results in a dictionary for dict-like access
        feature_results = {}

        for feature in self.features:
            if feature is None or feature in domain_strings:
                # Domain-based features return DataFrame
                cfg_file = tsfel.get_features_by_domain(feature)
                domain_df = tsfel.feature_extraction.calc_features.time_series_features_extractor(
                    cfg_file,
                    X,
                    **self.kwargs,
                )
                # Store domain result - use domain name as key
                feature_key = feature if feature is not None else "all"
                feature_results[feature_key] = domain_df
            else:
                # Individual features - store raw result
                feature_results[feature] = self._extract_individual_feature(X, feature)

        return _TSFELDataFrame([feature_results], index=[0])
