# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Individual TSFEL feature transformers."""

__author__ = ["Faakhir30"]

from sktime.transformations.series.tsfel._tsfel_feature_adapter import (
    _TSFELFeatureAdapter,
)


class AbsEnergyTransformer(_TSFELFeatureAdapter):
    """Computes the absolute energy of the signal."""

    def __init__(self):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.abs_energy,
            feature_name="abs_energy",
            output_type="Primitives",
        )


class AUCTransformer(_TSFELFeatureAdapter):
    """Area under curve feature transformer.

    Computes the area under the curve of the signal computed with trapezoid rule.

    Parameters
    ----------
    fs : float, optional (default=None)
        Sampling frequency. Required for this feature.

    Returns
    -------
    auc: float
        Area under the curve of the signal.

    """

    def __init__(self, fs=None):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.auc,
            feature_name="auc",
            output_type="Primitives",
            fs=fs,
        )


class AutocorrTransformer(_TSFELFeatureAdapter):
    """Autocorrelation feature transformer.

    Calculates the first 1/e crossing of the autocorrelation function (ACF).
    The adjusted ACF is calculated using the statsmodels.tsa.stattools.acf.
    Following the recommendations for long time series (size > 450), we use the FFT convolution.
    This feature measures the first time lag at which the autocorrelation function
    drops below 1/e (= 0.3679).

    Returns
    -------
    autocorr: int
        The first time lag at which the ACF drops below 1/e (= 0.3679).

    """

    def __init__(self):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.autocorr,
            feature_name="autocorr",
            output_type="Primitives",
        )


class AveragePowerTransformer(_TSFELFeatureAdapter):
    """Average power feature transformer.

    Computes the average power of the signal.

    Parameters
    ----------
    fs : float, optional (default=None)
        Sampling frequency. Required for this feature.

    Returns
    -------
    average_power: float
        Average power of the signal.
    """

    def __init__(self, fs=None):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.average_power,
            feature_name="average_power",
            output_type="Primitives",
            fs=fs,
        )


class CalcCentroidTransformer(_TSFELFeatureAdapter):
    """Temporal centroid feature transformer.

    Computes the centroid along the time axis.

    Parameters
    ----------
    fs : float, optional (default=None)
        Signal sampling frequency. Required for this feature.

    Returns
    -------
    centroid: float
        Centroid along the time axis.
    """

    def __init__(self, fs=None):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.calc_centroid,
            feature_name="calc_centroid",
            fs=fs,
        )


class CalcMaxTransformer(_TSFELFeatureAdapter):
    """Maximum value feature transformation estimator."""

    def __init__(self):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.calc_max,
            feature_name="calc_max",
            output_type="Primitives",
        )


class CalcMeanTransformer(_TSFELFeatureAdapter):
    """Mean value feature transformation estimator."""

    def __init__(self):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.calc_mean,
            feature_name="calc_mean",
            output_type="Primitives",
        )


class CalcMedianTransformer(_TSFELFeatureAdapter):
    """Median value feature transformation estimator."""

    def __init__(self):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.calc_median,
            feature_name="calc_median",
            output_type="Primitives",
        )


class CalcMinTransformer(_TSFELFeatureAdapter):
    """Minimum value feature transformation estimator."""

    def __init__(self):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.calc_min,
            feature_name="calc_min",
            output_type="Primitives",
        )


class CalcStdTransformer(_TSFELFeatureAdapter):
    """Standard deviation feature transformation estimator."""

    def __init__(self):
        import tsfel

        super().__init__(
            feature_func=tsfel.feature_extraction.features.calc_std,
            feature_name="calc_std",
            output_type="Primitives",
        )
