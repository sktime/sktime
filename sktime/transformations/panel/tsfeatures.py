"""TSFeatures transformer."""

__author__ = ["neha222222"]
__all__ = ["TSFeaturesTransformer"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class TSFeaturesTransformer(BaseTransformer):
    """TSFeatures feature extraction transformer.

    Direct interface to the ``tsfeatures`` package
    (https://github.com/Nixtla/tsfeatures).

    Extracts time series features commonly used for EDA and forecasting.

    Parameters
    ----------
    features : list of str or None, default=None
        List of feature names to extract. If None, extracts all default features.
    freq : int, default=1
        Frequency of the time series for seasonal features.
    scale : bool, default=True
        Whether to scale features to (0,1).

    Examples
    --------
    >>> from sktime.transformations.panel.tsfeatures import TSFeaturesTransformer
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> transformer = TSFeaturesTransformer()  # doctest: +SKIP
    >>> X_transformed = transformer.fit_transform(X_train)  # doctest: +SKIP

    References
    ----------
    .. [1] Hyndman, R.J., Wang, E., Laptev, N., et al. (2015).
           Large-scale unusual time series detection.
    """

    _tags = {
        "authors": ["neha222222"],
        "maintainers": ["neha222222"],
        "python_dependencies": "tsfeatures",
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "capability:multivariate": False,
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "tests:vm": True,
    }

    def __init__(self, features=None, freq=1, scale=True):
        self.features = features
        self.freq = freq
        self.scale = scale
        super().__init__()

    def _transform(self, X, y=None):
        import tsfeatures

        df = pd.DataFrame({
            "unique_id": [1] * len(X),
            "ds": pd.date_range(start="2020-01-01", periods=len(X)),
            "y": X.values,
        })

        kwargs = {"freq": self.freq, "scale": self.scale}
        if self.features is not None:
            kwargs["features"] = self.features

        result = tsfeatures.tsfeatures(df, **kwargs)

        result = result.drop(columns=["unique_id"], errors="ignore")
        return result

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {"features": None, "freq": 1, "scale": False},
            {"features": ["hurst", "stability"], "freq": 1, "scale": True},
        ]

