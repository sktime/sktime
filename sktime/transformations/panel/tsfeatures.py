# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TSFeatures transformer."""

__author__ = ["Faakhir30"]
__all__ = ["TSFeaturesTransformer"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class TSFeaturesTransformer(BaseTransformer):
    """Transformer for extracting time series features via tsfeatures.

    Direct interface to tsfeatures.tsfeatures [1] as an sktime transformer.
    This transformer works with Series with 1 column and a datetime index.

    By default, this transformer uses 17 feature functions that extract approximately
    34 features (42 for seasonal data with freq > 1). The default feature functions
    include:

    **Autocorrelation Features**:

    - ``acf_features:`` Autocorrelation function features (6-7 features)
    - ``pacf_features:`` Partial autocorrelation function features (3-4 features)

    **Model-based Features**:

    - ``arch_stat:`` ARCH model test statistic
    - ``heterogeneity:`` ARCH/GARCH heterogeneity features (4 features)
    - ``holt_parameters:`` Holt exponential smoothing parameters (2 features)
    - ``hw_parameters:`` Holt-Winters parameters (3 features, only if freq > 1)
    - ``stl_features:`` Seasonal-trend decomposition features (8-11 features)

    **Statistical Features**:

    - ``crossing_points:`` Number of median crossings
    - ``entropy:`` Spectral entropy
    - ``flat_spots:`` Number of flat spots
    - ``hurst:`` Hurst exponent
    - ``lumpiness:`` Variance of variances across windows
    - ``nonlinearity:`` Terasvirta nonlinearity test
    - ``stability:`` Variance of means across windows
    - ``unitroot_kpss:`` KPSS unit root test statistic
    - ``unitroot_pp:`` Phillips-Perron unit root test statistic

    **Basic Features**:

    - ``series_length:`` Length of the time series

    **Other supported features (non-default)**:

    - ``count_entropy:`` Entropy using only positive data.
    - ``intervals:`` Mean and Standard Deviation of intervals with positive values.
    - ``frequency:`` Wrapper of freq parameter.
    - ``guerrero:`` Applies Guerrero's (1993) method to select the lambda which
    minimises the coefficient of variation for subseries of x.
    - ``sparsity:`` Average obs with zero values.

    Parameters
    ----------
    features : list of callable, optional
        List of feature functions to compute. If None, uses default feature set.
    scale : bool, optional (default=True)
        Whether to (mean-std) scale data before computing features.

    References
    ----------
    .. [1] https://github.com/Nixtla/tsfeatures

    Examples
    --------
    >>> from sktime.transformations.panel.tsfeatures import TSFeaturesTransformer
    >>> from sktime.utils._testing.series import _make_series
    >>> X = _make_series()
    >>> transformer = TSFeaturesTransformer()
    >>> Xt = transformer.fit_transform(X)
    >>> # Example using specific features
    >>> from tsfeatures.tsfeatures import acf_features # doctest: +SKIP
    >>> acf_transformer = TSFeaturesTransformer(
    ...    features=[acf_features],
    ...    ) # doctest: +SKIP
    >>> acf_Xt = acf_transformer.fit_transform(X) # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "FedericoGarza",
            "kdgutier",
            "cristianchallu",
            "jose-moralez",
            "rolivaresar",
            "mergenthaler",
        ],
        "maintainers": ["Faakhir30"],
        "python_dependencies": ["tsfeatures"],
        #
        # estimator type
        # ----------------------------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "fit_is_empty": True,
        #
        # behavioural tags: internal type
        # ----------------------------------
        "requires_y": False,
        "y_inner_mtype": "None",
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        "capability:unequal_length": True,
        "capability:missing_values": False,
        "capability:categorical_in_X": False,
        "capability:multivariate": False,
        # testing configuration
        # ---------------------
        "tests:vm": True,
    }

    def __init__(
        self,
        features=None,
        scale=True,
    ):
        self.features = features
        self.scale = scale
        super().__init__()

    def _transform(self, X, y=None):
        """Extract time series features for a single univariate time series.

        This private method performs the core logic to convert a single univariate
        time series into a DataFrame of extracted features using the `tsfeatures`
        library. It is called indirectly via the public `transform` interface.

        Parameters
        ----------
        X : Series
            A single univariate time series with 1 column.
            If the index is not a DatetimeIndex, a default daily DatetimeIndex
            will be created.
        y : None
            Ignored. Present only for interface compatibility.

        Returns
        -------
        Xt : pd.DataFrame
            A single-row DataFrame containing extracted features. The columns
            correspond to feature names from `tsfeatures`. The DataFrame has one
            row for the input time series.
        """
        from tsfeatures import tsfeatures as tsfeatures_func
        from tsfeatures.utils import FREQS

        features_to_use = (
            self.features if self.features is not None else self._get_default_features()
        )

        # Ensure X is a Series
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                X = X.iloc[:, 0]
            else:
                raise ValueError(
                    f"X must be univariate (1 column), but got {X.shape[1]} columns"
                )

        values = X.values
        # Handle non-DatetimeIndex
        if not isinstance(X.index, pd.DatetimeIndex):
            datetime_index = pd.date_range(start="2000-01-01", periods=len(X), freq="D")
        else:
            datetime_index = X.index

        # Convert to tsfeatures long format ['unique_id', 'ds', 'y']
        ts_long = pd.DataFrame(
            {
                # used "0" as arbitrary unique_id
                # since considering X as a single Series
                "unique_id": "0",
                "ds": datetime_index,
                "y": values,
            }
        )

        freq = FREQS.get(pd.infer_freq(datetime_index))

        # Extract features
        Xt = tsfeatures_func(
            ts=ts_long,
            freq=freq,
            features=features_to_use,
            scale=self.scale,
        )

        # Remove unique_id column if present
        if "unique_id" in Xt.columns:
            Xt = Xt.drop(columns=["unique_id"])

        return Xt

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
        from sktime.utils.dependencies import _check_soft_dependencies

        params = [
            {
                "features": None,
                "scale": True,
            },
            {
                "scale": False,
            },
        ]

        if _check_soft_dependencies("tsfeatures", severity="none"):
            from tsfeatures.tsfeatures import acf_features, arch_stat

            params.append(
                {
                    "features": [acf_features, arch_stat],
                    "scale": True,
                }
            )

        return params
