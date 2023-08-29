"""Pipeline making utility."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]


def make_pipeline(*steps):
    """Create a pipeline from estimators of any type.

    Parameters
    ----------
    steps : tuple of sktime estimators
        in same order as used for pipeline construction

    Returns
    -------
    pipe : sktime pipeline containing steps, in order
        always a descendant of BaseObject, precise object determined by scitype
        equivalent to result of step[0] * step[1] * ... * step[-1]

    Examples
    --------
    Example 1: forecaster pipeline

    >>> from sktime.pipeline import make_pipeline
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> pipe = make_pipeline(ExponentTransformer(), PolynomialTrendForecaster())
    >>> type(pipe).__name__
    'TransformedTargetForecaster'

    Example 2: classifier pipeline

    >>> from sktime.pipeline import make_pipeline
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    >>> pipe = make_pipeline(ExponentTransformer(), KNeighborsTimeSeriesClassifier())
    >>> type(pipe).__name__
    'ClassifierPipeline'

    Example 3: transformer pipeline

    >>> from sktime.pipeline import make_pipeline
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> pipe = make_pipeline(ExponentTransformer(), ExponentTransformer())
    >>> type(pipe).__name__
    'TransformerPipeline'
    """
    pipe = steps[0]
    for i in range(1, len(steps)):
        pipe = pipe * steps[i]

    return pipe
