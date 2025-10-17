"""Forecasting Datasets."""

__author__ = ["jgyasu"]
__all__ = ["ForecastingData"]

import warnings
from inspect import signature

import pandas as pd

from sktime.datasets import load_forecastingdata
from sktime.datasets.base._base import InvalidSetError
from sktime.datasets.forecasting._base import BaseForecastingDataset
from sktime.datasets.forecasting.monash._tags import DATASET_TAGS

FORCE_RANGEINDEX = {
    "m4_daily_dataset",
    "m4_hourly_dataset",
    "m4_monthly_dataset",
    "m4_quarterly_dataset",
    "m4_weekly_dataset",
    "m4_yearly_dataset",
}


class ForecastingData(BaseForecastingDataset):
    """Monash Forecasting Repository dataset loader.

    Generic loader for forecasting datasets from the Monash Time Series
    Forecasting Repository. Provides access to benchmark datasets widely
    used in forecasting research and competitions, including domains such
    as finance, tourism, healthcare, energy, and retail.

    Supports loading both target series (y) and optional exogenous variables (X).
    Data are returned in sktime in-memory data representation.

    Examples
    --------
    >>> from sktime.datasets import ForecastingData # doctest: +SKIP
    >>> dataset = ForecastingData(name="cif_2016_dataset") # doctest: +SKIP
    >>> y = dataset.load("y") # doctest: +SKIP

    Notes
    -----
    Dimensionality: univariate or multivariate (depends on dataset)
    Frequency: yearly, quarterly, monthly, daily, etc. (varies)
    Exogenous data: optional, available for some datasets

    Dataset details: https://forecastingdata.org
    """

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.loader_func = load_forecastingdata

        self.set_tags(name=self.name)
        self.set_tags(**DATASET_TAGS[self.name])

    def _encode_args(self, code):
        """Decide kwargs for the loader function (no splits in forecasting datasets)."""
        kwargs = {}

        if code not in ["X", "y"]:
            raise InvalidSetError(
                f"Dataset {self.name} does not define a '{code}' set. "
                "Use 'X' or 'y' and perform your own temporal split."
            )

        loader_func_params = signature(self.loader_func).parameters
        init_signature_params = signature(self.__init__).parameters
        init_param_values = {k: getattr(self, k) for k in init_signature_params.keys()}

        for init_param_name, init_param_value in init_param_values.items():
            if init_param_name in loader_func_params:
                kwargs[init_param_name] = init_param_value

        return kwargs

    def _load(self, *args):
        """Load the dataset.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            "X": exogeneous time series
            "y": time series

        Returns
        -------
        dataset, if args is empty or length one
            data container corresponding to string in args
        tuple, of same length as args, if args is length 2 or longer
        """
        if len(args) == 0:
            args = ("X", "y")

        cache = {}

        if "X" in args or "y" in args:
            X, y = self._load_dataset(**self._encode_args("X"))
            cache["X"] = X
            cache["y"] = y

        for code in args:
            if code not in cache:
                raise InvalidSetError(
                    f"Dataset {self.name} does not provide a '{code}' split."
                )

        res = [cache[key] for key in args]
        return res[0] if len(res) == 1 else tuple(res)

    def _load_dataset(self, **kwargs):
        """Call loader function with self.name included automatically."""
        if "name" in signature(self.loader_func).parameters and "name" not in kwargs:
            kwargs["name"] = self.name
        dataset, metadata = self.loader_func(**kwargs)
        y, X = self._split_into_y_and_X(dataset)
        if metadata["frequency"] is not None:
            y = self._to_sktime_multiindex(y, metadata)
        if self.name in FORCE_RANGEINDEX:
            warnings.warn(
                "Due to errors in timestamps of the original dataset, "
                "RangeIndex is used to construct the timepoints."
            )
        return X, y

    def _to_sktime_multiindex(self, y, metadata):
        """Convert dataset into sktime pd-multiindex format with correct time index."""
        freq_map = {
            "yearly": "YS",
            "quarterly": "QS",
            "monthly": "MS",
            "weekly": "W",
            "daily": "D",
            "hourly": "H",
            "half_hourly": (30, "min"),
            "10_minutes": (10, "min"),
            "minutely": (1, "min"),
            "4_seconds": (4, "s"),
            None: None,
        }

        freq_label = metadata.get("frequency")
        if freq_label is not None:
            freq_label = freq_label.lower()
        freq = freq_map.get(freq_label)
        if freq is None:
            raise ValueError(f"Unknown frequency label: {freq_label}")

        has_start = "start_timestamp" in y.columns

        records = []
        for _, row in y.iterrows():
            series_id = row["series_name"]
            values = row["series_value"]
            n = values.size

            if self.name in FORCE_RANGEINDEX:
                time_index = pd.RangeIndex(start=0, stop=n, step=1)
            elif has_start:
                start = pd.to_datetime(row["start_timestamp"])
                if isinstance(freq, str):
                    time_index = pd.date_range(start=start, periods=n, freq=freq)
                else:
                    step, unit = freq
                    time_index = start + pd.timedelta_range(
                        start=0, periods=n, freq=f"{step}{unit}"
                    )
            else:
                time_index = pd.RangeIndex(start=0, stop=n, step=1)

            records.extend((series_id, t, v) for t, v in zip(time_index, values))

        panel = pd.DataFrame.from_records(
            records, columns=["instances", "timepoints", "value"]
        )
        panel = panel.set_index(["instances", "timepoints"])
        return panel

    def _split_into_y_and_X(self, loader_output):
        """Split the output of the loader into X and y."""
        if isinstance(loader_output, tuple):
            return loader_output
        y = loader_output
        X = None
        return (y, X)

    def all_datasets():
        """Return list of dataset names loadable through this class."""
        return list(DATASET_TAGS.keys())

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter."""
        params_list = [
            {
                "name": "cif_2016_dataset",
            },
            {
                "name": "hospital_dataset",
            },
        ]

        return params_list
