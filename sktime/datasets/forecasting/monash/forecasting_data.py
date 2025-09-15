"""Forecasting Datasets."""

__author__ = ["jgyasu"]
__all__ = ["ForecastingData"]

import warnings
from inspect import signature

import pandas as pd

from sktime.datasets import load_forecastingdata
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
    """Forecasting dataset loader.

    Examples
    --------
    >>> from sktime.datasets import ForecastingData
    >>> dataset = ForecastingData(name="cif_2016_dataset")
    >>> y = dataset.load("y")
    """

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.loader_func = load_forecastingdata

        self.set_tags(name=self.name)
        self.set_tags(**DATASET_TAGS[self.name])

    def _encode_args(self, code):
        """Decide kwargs for the loader function."""
        kwargs = {}

        if code in ["X", "y"]:
            split = None
        elif code in ["X_train", "y_train"]:
            split = "TRAIN"
        elif code in ["X_test", "y_test"]:
            split = "TEST"
        else:
            split = None

        loader_func_params = signature(self.loader_func).parameters
        init_signature_params = signature(self.__init__).parameters
        init_param_values = {k: getattr(self, k) for k in init_signature_params.keys()}

        if (
            "test" in code.lower() or "train" in code.lower()
        ) and "split" not in loader_func_params:
            raise ValueError(
                "This dataset loader does not have a train/test split. "
                "Load the full dataset instead."
            )

        if "split" in loader_func_params:
            kwargs["split"] = split

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
            "X_train": training instances only, for fixed single split
            "y_train": training labels only, for fixed single split
            "X_test": test instances only, for fixed single split
            "y_test": test labels only, for fixed single split

        Returns
        -------
        dataset, if args is empty or length one
            data container corresponding to string in args (see above)
        tuple, of same length as args, if args is length 2 or longer
            data containers corresponding to strings in args, in same order
        """
        if len(args) == 0:
            args = ("X", "y")

        cache = {}

        if "X" in args or "y" in args:
            X, y = self._load_dataset(**self._encode_args("X"))
            cache["X"] = X
            cache["y"] = y
        if "X_train" in args or "y_train" in args:
            X, y = self._load_dataset(**self._encode_args("X_train"))
            cache["X_train"] = X
            cache["y_train"] = y
        if "X_test" in args or "y_test" in args:
            X, y = self._load_dataset(**self._encode_args("X_test"))
            cache["X_test"] = X
            cache["y_test"] = y
        if "cv" in args:
            cv = self._load_simple_train_test_cv_split()
            cache["cv"] = cv

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
        """
        Convert dataset into sktime pd-multiindex format with correct time index.

        Parameters
        ----------
        y : pd.DataFrame
            Must contain "series_name" and "series_value".
            Optionally, "start_timestamp".
        metadata : dict

        Returns
        -------
        pd.DataFrame in sktime pd-multiindex format
        """
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
                    # calendar-based frequency
                    time_index = pd.date_range(start=start, periods=n, freq=freq)
                else:
                    # fixed offset frequency
                    step, unit = freq
                    time_index = start + pd.timedelta_range(
                        start=0, periods=n, freq=f"{step}{unit}"
                    )
            else:
                # default RangeIndex if no start timestamp given
                time_index = pd.RangeIndex(start=0, stop=n, step=1)

            records.extend((series_id, t, v) for t, v in zip(time_index, values))

        panel = pd.DataFrame.from_records(
            records, columns=["instances", "timepoints", "value"]
        )
        panel = panel.set_index(["instances", "timepoints"])
        return panel

    def _split_into_y_and_X(self, loader_output):
        """Split the output of the loader into X and y.

        Parameters
        ----------
        loader_output: any
            Output of the loader function.

        Returns
        -------
        tuple
            Tuple containing y and X.
        """
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
