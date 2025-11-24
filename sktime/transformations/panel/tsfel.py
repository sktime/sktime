"""TSFEL features.

A transformer for the TSFEL features.
"""

__author__ = ["neha222222"]
__all__ = ["TSFELTransformer"]

from sktime.transformations.base import BaseTransformer


class TSFELTransformer(BaseTransformer):
    """Time Series Feature Extraction Library (TSFEL) features.

    Direct interface to the ``tsfel`` implementation of TSFEL feature extraction
    (https://github.com/fraunhoferportugal/tsfel).

    Extracts time series features across multiple domains: statistical, temporal,
    spectral, and fractal.

    Parameters
    ----------
    features : str or dict, optional, default="minimal"
        Which features to extract.
        If "minimal", extracts a minimal set of features (first 3 from each domain).
        If "all", extracts all available features.
        If dict, custom feature configuration dictionary in tsfel format.
    fs : int, optional, default=100
        Sampling frequency of the signal.
    verbose : int, optional, default=0
        Level of verbosity. 0 = silent, 1 = progress bar.

    Examples
    --------
    >>> from sktime.transformations.panel.tsfel import TSFELTransformer
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> transformer = TSFELTransformer(features="minimal")  # doctest: +SKIP
    >>> X_transformed = transformer.fit_transform(X_train)  # doctest: +SKIP

    References
    ----------
    .. [1] Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M.,
           Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020).
           TSFEL: Time Series Feature Extraction Library. SoftwareX, 11, 100456.
    """

    _tags = {
        "authors": ["neha222222"],
        "maintainers": ["neha222222"],
        "python_dependencies": "tsfel",
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "capability:multivariate": False,
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "tests:vm": True,
    }

    def __init__(self, features="minimal", fs=100, verbose=0):
        self.features = features
        self.fs = fs
        self.verbose = verbose
        super().__init__()

    def _transform(self, X, y=None):
        import tsfel

        if self.features == "minimal":
            cfg = tsfel.get_features_by_domain()
            cfg_file = {d: dict(list(f.items())[:3]) for d, f in cfg.items()}
        elif self.features == "all":
            cfg_file = tsfel.get_features_by_domain()
        elif isinstance(self.features, dict):
            cfg_file = self.features
        else:
            raise ValueError("features must be 'minimal', 'all', or dict")

        X_df = X.to_frame()
        return tsfel.time_series_features_extractor(
            cfg_file, X_df, fs=self.fs, verbose=self.verbose
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"features": "minimal", "fs": 100}, {"features": "all", "fs": 50}]

