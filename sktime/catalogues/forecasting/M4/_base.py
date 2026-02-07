"""M4 forecasting competition catalogues by period."""

from sktime.catalogues.base import BaseCatalogue


class _BaseM4CompetitionCatalogue(BaseCatalogue):
    """Base catalogue for M4 forecasting competition benchmarks.

    This base class defines the common structure, forecasters, and evaluation
    protocol used to reproduce classical statistical and machine learning
    benchmarks from the M4 forecasting competition.

    Concrete subclasses represent a single temporal granularity of the M4
    competition (hourly, daily, weekly, monthly, quarterly, or yearly) and
    bind the corresponding dataset and seasonal period (sp) used by the
    Overall Weighted Average (OWA) metric.

    The M4 competition is a large-scale forecasting benchmark that evaluates
    forecast accuracy across multiple temporal granularities. This catalogue
    family exposes commonly used classical baselines from the M4 literature
    in a unified, reproducible form compatible with the sktime benchmarking
    framework.
    """

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "object_type": "catalogue",
        "catalogue_type": "mixed",
        "n_items": 10,
        "n_datasets": 0,
        "n_forecasters": 10,
        "n_metrics": 0,
        "n_cv_splitters": 0,
    }

    _dataset_name = None
    _metric_name = None
    _base_forecasters = [
        ("Naive_1", "NaiveForecaster(strategy='last')"),
        ("SES", "ExponentialSmoothing(trend=None, seasonal=None)"),
        ("Holt", "ExponentialSmoothing(trend='add', seasonal=None)"),
        ("Damped", "ExponentialSmoothing(trend='add', damped_trend=True)"),
        ("Theta", "ThetaForecaster()"),
        ("AutoARIMA", "AutoARIMA()"),
        ("AutoETS", "AutoETS()"),
        (
            "Comb",
            "EnsembleForecaster("
            "    ["
            "        ('ses', ExponentialSmoothing(trend=None)),"
            "        ('holt', ExponentialSmoothing(trend='add')),"
            "        ('damped', ExponentialSmoothing(trend='add', damped_trend=True)),"
            "    ],"
            "    aggfunc='mean',"
            ")",
        ),
    ]

    _specific_forecasters = []

    def _get(self):
        """Return a dict of items (datasets, forecasters, metrics)."""
        if self._dataset_name is None or self._metric_name is None:
            raise ValueError("_dataset_name and _metric_name must be set in subclass")

        forecasters = self._base_forecasters + self._specific_forecasters

        return {
            "dataset": [f"ForecastingData('{self._dataset_name}')"],
            "forecaster": forecasters,
            "metric": [f"{self._metric_name}"],
            "cv_splitter": ["TemporalTrainTestSplitter(test_size=0.2)"],
        }
