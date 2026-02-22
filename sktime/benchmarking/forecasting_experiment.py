"""Experiment adaptor for forecasting cv experiments."""

from sktime.benchmarking.base import BaseExperiment


class SktimeForecastingExperiment(BaseExperiment):
    """Experiment adaptor for forecasting cross-validation experiments.

    This class is used to perform cross-validation experiments using a given
    sktime forecaster. It allows for hyperparamter tuning and evaluation of the
    forecaster's performance using cross-validation.
    """
