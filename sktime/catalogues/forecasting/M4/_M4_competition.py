"""M4 forecasting competition catalogue."""

from sktime.catalogues.base import BaseCatalogue


class M4CompetitionCatalogue(BaseCatalogue):
    """M4 forecasting competition catalogue.

    Catalogue of datasets, forecasters, metrics, and cross-validation splitters
    used to reproduce classical statistical and ML benchmarks from the M4 forecasting
    competition.

    The M4 competition is a large-scale forecasting benchmark that evaluates
    forecast accuracy across multiple temporal granularities, including hourly,
    daily, weekly, monthly, quarterly, and yearly time series. This catalogue
    collects commonly used classical baselines from the M4 literature and exposes
    them in a unified, reproducible form compatible with the sktime benchmarking
    framework.
    """

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "object_type": "catalogue",
        "catalogue_type": "mixed",
        "n_items": 18,
        "n_datasets": 6,
        "n_forecasters": 11,
        "n_metrics": 1,
        "n_cv_splitters": 0,
    }

    def _get(self):
        """Return a dict of items (datasets, forecasters, metrics)."""
        datasets = [
            "m4_hourly_dataset",
            "m4_daily_dataset",
            "m4_weekly_dataset",
            "m4_monthly_dataset",
            "m4_quarterly_dataset",
            "m4_yearly_dataset",
        ]

        forecasters = [
            "NaiveForecaster(strategy='last')",  # Naïve 1
            "NaiveForecaster(strategy='last', sp=1)",  # Naïve S
            "ExponentialSmoothing(trend=None, seasonal=None)",  # SES
            "ExponentialSmoothing(trend='add', seasonal=None)",  # Holt
            "ExponentialSmoothing(trend='add', damped_trend=True)",  # Damped
            "ThetaForecaster()",  # Theta
            "AutoARIMA()",  # ARIMA
            "AutoETS()",  # ETS
            "EnsembleForecaster("
            "    ["
            "        ('ses', ExponentialSmoothing(trend=None)),"
            "        ('holt', ExponentialSmoothing(trend='add')),"
            "        ('damped', ExponentialSmoothing(trend='add', damped_trend=True)),"
            "    ],"
            "    aggfunc='mean',"
            ")",  # Comb
        ]

        metrics = ["OverallWeightedAverage()"]

        cv_splitters = [...]

        all_objects = {
            "dataset": [f"ForecastingData('{dataset}')" for dataset in datasets],
            "forecaster": forecasters,
            "metric": metrics,
            "cv_splitter": cv_splitters,
        }
        return all_objects
