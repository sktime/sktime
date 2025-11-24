# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TSFeatures transformer."""

__author__ = ["Faakhir30"]
__all__ = ["TSFeaturesTransformer", "TSFeaturesWideTransformer"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class _TSFeaturesBase(BaseTransformer):
    """Base class for TSFeatures transformers.

    This base class contains common functionality shared between
    TSFeaturesTransformer (long format) and TSFeaturesWideTransformer (wide format).
    """

    _tags = {
        "authors": ["Faakhir30"],
        "maintainers": ["Faakhir30"],
        "python_dependencies": ["tsfeatures"],
        #
        # estimator type
        # ----------------------------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        #
        # behavioural tags: internal type
        # ----------------------------------
        "X_inner_mtype": "nested_univ",
        "requires_y": False,
        "y_inner_mtype": "None",
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        "fit_is_empty": True,
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
        scale=True,
        threads=None,
    ):
        super().__init__()
        self.features = features
        self.scale = scale
        self.threads = threads

    def _get_default_features(self):
        """Get default feature functions from tsfeatures."""
        from tsfeatures.tsfeatures import (
            acf_features,
            arch_stat,
            crossing_points,
            entropy,
            flat_spots,
            heterogeneity,
            holt_parameters,
            hurst,
            hw_parameters,
            lumpiness,
            nonlinearity,
            pacf_features,
            series_length,
            stability,
            stl_features,
            unitroot_kpss,
            unitroot_pp,
        )
        return [
            acf_features,
            arch_stat,
            crossing_points,
            entropy,
            flat_spots,
            heterogeneity,
            holt_parameters,
            lumpiness,
            nonlinearity,
            pacf_features,
            stl_features,
            stability,
            hw_parameters,
            unitroot_kpss,
            unitroot_pp,
            series_length,
            hurst,
        ]


class TSFeaturesTransformer(_TSFeaturesBase):
    """Transformer for extracting time series features via tsfeatures (long format).

    Direct interface to tsfeatures.tsfeatures [1] as an sktime transformer.
    This transformer works with long format data where each time series is
    represented as ['unique_id', 'ds', 'y'].

    By default, this transformer uses 17 feature functions that extract approximately
    34 features (42 for seasonal data with freq > 1). The default feature functions
    include:

    **Autocorrelation Features**:
        - acf_features: Autocorrelation function features (6-7 features)
        - pacf_features: Partial autocorrelation function features (3-4 features)

    **Model-based Features**:
        - arch_stat: ARCH model test statistic
        - heterogeneity: ARCH/GARCH heterogeneity features (4 features)
        - holt_parameters: Holt exponential smoothing parameters (2 features)
        - hw_parameters: Holt-Winters parameters (3 features, only if freq > 1)
        - stl_features: Seasonal-trend decomposition features (8-11 features)

    **Statistical Features**:
        - crossing_points: Number of median crossings
        - entropy: Spectral entropy
        - flat_spots: Number of flat spots
        - hurst: Hurst exponent
        - lumpiness: Variance of variances across windows
        - nonlinearity: Terasvirta nonlinearity test
        - stability: Variance of means across windows
        - unitroot_kpss: KPSS unit root test statistic
        - unitroot_pp: Phillips-Perron unit root test statistic

    **Basic Features**:
        - series_length: Length of the time series

    Parameters
    ----------
    freq : int, optional (default=None)
        Frequency of the time series. If None, the frequency of each time series
        is inferred and seasonal periods are assigned according to dict_freqs.
    features : list of callable, optional
        List of feature functions to compute. If None, uses default feature set.
    dict_freqs : dict, optional (default=tsfeatures.FREQS)
        Dictionary that maps string frequency to int. Example: {'D': 7, 'W': 1}
    scale : bool, optional (default=True)
        Whether to (mean-std) scale data before computing features.
    threads : int, optional (default=None)
        Number of threads to use. Use None (default) for parallel processing.

    References
    ----------
    .. [1] https://github.com/Nixtla/tsfeatures

    Examples
    --------
    >>> from sktime.datasets import load_arrow_head
    >>> from sktime.transformations.panel.tsfeatures import TSFeaturesTransformer
    >>> X, y = load_arrow_head(return_X_y=True)
    >>> transformer = TSFeaturesTransformer(freq=1, scale=True)
    >>> Xt = transformer.fit_transform(X)
    """

    def __init__(
        self,
        freq=None,
        features=None,
        dict_freqs=None,
        scale=True,
        threads=None,
    ):
        super().__init__(features=features, scale=scale, threads=threads)
        self.freq = freq
        self.dict_freqs = dict_freqs


    def _transform(self, X, y=None):
        """Extract time series features for each instance in X.

        This private method performs the core logic to convert a nested DataFrame
        (where each cell contains a pd.Series representing a time series) into a
        DataFrame of extracted features using the `tsfeatures` library. It is
        called indirectly via the public `transform` interface.

        Parameters
        ----------
        X : pd.DataFrame
            A nested pandas DataFrame of shape [n_instances, n_features],
            where each cell contains a pd.Series representing a time series to
            extract features from.
        y : None
            Ignored. Present only for interface compatibility.

        Returns
        -------
        Xt : pd.DataFrame
            A DataFrame containing extracted features for each instance
            (row). The number of columns corresponds to the total number of
            features extracted by `tsfeatures`. The DataFrame has one row per
            input instance.
        """
        from tsfeatures import tsfeatures as tsfeatures_func

        features_to_use = (
            self.features
            if self.features is not None
            else self._get_default_features()
        )

        dict_freqs_to_use = self.dict_freqs
        if dict_freqs_to_use is None:
            from tsfeatures.utils import FREQS
            dict_freqs_to_use = FREQS

        # Convert nested_univ to long format ['unique_id', 'ds', 'y']
        ts_long_list = []
        instance_to_uid = {}
        for instance_idx in X.index:
            series = X.loc[instance_idx, X.columns[0]]
            if isinstance(series, pd.Series):
                unique_id = str(instance_idx)
                instance_to_uid[instance_idx] = unique_id
                ts_long_list.append(pd.DataFrame({
                    'unique_id': unique_id,
                    'ds': series.index.values,
                    'y': series.values
                }))

        ts_long = (
            pd.concat(ts_long_list, ignore_index=True)
            if ts_long_list
            else pd.DataFrame(columns=['unique_id', 'ds', 'y'])
        )

        Xt = tsfeatures_func(
            ts=ts_long,
            freq=self.freq,
            features=features_to_use,
            dict_freqs=dict_freqs_to_use,
            scale=self.scale,
            threads=self.threads,
        )

        # Map back to original indices
        if 'unique_id' in Xt.columns:
            uid_to_instance = {v: k for k, v in instance_to_uid.items()}
            Xt = Xt.set_index('unique_id')
            Xt.index = Xt.index.map(uid_to_instance)
            Xt = Xt.reindex(X.index)

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from tsfeatures.tsfeatures import acf_features, arch_stat
        return [
            {
                "freq": 1,
                "dict_freqs": None,
                "scale": True,
                "threads": 1,
            },
            {
                "scale": False,
                "threads": None,
                "freq": 1,
            },
            {
                "features": [acf_features, arch_stat],
                "scale": True,
                "threads": 1,
                "freq": 2,
            },
        ]


class TSFeaturesWideTransformer(_TSFeaturesBase):
    """Transformer for extracting time series features via tsfeatures (wide format).

    Direct interface to tsfeatures.tsfeatures_wide [1] as an sktime transformer.
    This transformer works with wide format data where each time series is
    represented as ['unique_id', 'seasonality', 'y'] where 'y' is an array.

    By default, this transformer uses 17 feature functions that extract approximately
    34 features (42 for seasonal data). The default feature functions are the same
    as TSFeaturesTransformer (see its docstring for details).

    Parameters
    ----------
    features : list of callable, optional
        List of feature functions to compute. If None, uses default feature set.
    scale : bool, optional (default=True)
        Whether to (mean-std) scale data before computing features.
    threads : int, optional (default=None)
        Number of threads to use. Use None (default) for parallel processing.

    References
    ----------
    .. [1] https://github.com/Nixtla/tsfeatures

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sktime.transformations.panel.tsfeatures import TSFeaturesWideTransformer
    >>> # Create wide format data
    >>> data = pd.DataFrame({
    ...     'unique_id': [0, 1],
    ...     'seasonality': [12, 4],
    ...     'y': [np.random.randn(100), np.random.randn(50)]
    ... })
    >>> transformer = TSFeaturesWideTransformer(scale=True)
    >>> Xt = transformer.fit_transform(data)
    """

    def __init__(self, features=None, scale=True, threads=None):
        super().__init__(features=features, scale=scale, threads=threads)
        self.set_tags(
            **{
                "X_inner_mtype": "pd.DataFrame",
                "capability:categorical_in_X": True,
                "tests:skip_by_name": [
                    "test_fit_idempotent",
                    "test_methods_have_no_side_effects",
                    "test_non_state_changing_method_contract",
                    "test_persistence_via_pickle",
                    "test_save_estimators_to_file",
                    "test_categorical_y_raises_error",
                    "test_categorical_X_passes",
                    "test_fit_transform_output",
                ],
            }
        )
    def _transform(self, X, y=None):
        """Extract time series features for each instance in X (wide format).

        This private method performs the core logic to convert wide format data
        (with ['unique_id', 'seasonality', 'y'] columns) into a DataFrame of
        extracted features using the `tsfeatures.tsfeatures_wide` function.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame with columns ['unique_id', 'seasonality', 'y'],
            where 'y' contains arrays representing time series.
        y : None
            Ignored. Present only for interface compatibility.

        Returns
        -------
        Xt : pd.DataFrame
            A DataFrame containing extracted features for each instance (row).
            The number of columns corresponds to the total number of features
            extracted by `tsfeatures`. The DataFrame has one row per input instance.
        """
        from tsfeatures import tsfeatures_wide as tsfeatures_wide_func

        required_cols = ['unique_id', 'seasonality', 'y']
        if not all(col in X.columns for col in required_cols):
            raise ValueError(
                f"X must have columns {required_cols}. "
                f"Found columns: {list(X.columns)}"
            )

        features_to_use = (
            self.features if self.features is not None else self._get_default_features()
        )

        Xt = tsfeatures_wide_func(
            ts=X,
            features=features_to_use,
            scale=self.scale,
            threads=self.threads,
        )

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from tsfeatures.tsfeatures import acf_features, arch_stat
        return [
            {
                "scale": True,
                "threads": 4,
            },
            {
                "features": [acf_features],
                "scale": False,
                "threads": None,
            },
            {
                "features": [acf_features, arch_stat],
                "scale": True,
                "threads": 1,
            },
        ]
