# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TSFEL transformer to extract features by domain or specific feature names."""

__author__ = ["Faakhir30"]


from sktime.transformations.base import BaseTransformer


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
        "scitype:transform-output": "Series",
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
        self.kwargs = kwargs

        super().__init__()

    def _extract_individual_feature(self, X, feature_name):
        """Extract a single feature by calling individual feature functions directly.

        Analyzes the feature function signature and passes appropriate parameters
        from kwargs. Returns the result directly from the feature function.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Input data
        feature_name : str
            Name of the feature function to extract (e.g., 'abs_energy', 'auc')

        Returns
        -------
        pd.DataFrame or pd.Series or scalar
            Result from the feature function. Format depends on the feature function.
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

        feature_kwargs = {}

        for param_name, param in sig_params.items():
            if param_name == "signal":
                continue
            if param_name in self.kwargs:
                feature_kwargs[param_name] = self.kwargs[param_name]
            elif param.default != Parameter.empty:
                feature_kwargs[param_name] = param.default

        return feature_func(X, **feature_kwargs)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        X_transformed : Series of mtype pd.DataFrame
            transformed version of X containing extracted features
        """
        import tsfel
        import pandas as pd

        domain_strings = ["statistical", "temporal", "spectral", "fractal"]

        result_dfs = []

        for feature in self.features:
            if feature is None or feature in domain_strings:
                cfg_file = tsfel.get_features_by_domain(feature)
                domain_df = tsfel.feature_extraction.calc_features.time_series_features_extractor(
                    cfg_file,
                    X,
                    **self.kwargs,
                )
                result_dfs.append(domain_df)
            else:
                individual_df = self._extract_individual_feature(X, feature)
                result_dfs.append(individual_df)

        # Combine all results
        if result_dfs:
            return pd.concat(result_dfs, axis=1)
