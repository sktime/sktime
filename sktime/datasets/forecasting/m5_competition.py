"""M5 competition dataset."""

from pathlib import Path

import pandas as pd

from sktime.datasets._data_io import _download_and_extract, _reduce_memory_usage
from sktime.datasets.forecasting._base import BaseForecastingDataset


class M5Dataset(BaseForecastingDataset):
    """Fetch M5 dataset from https://zenodo.org/records/12636070 .

    Downloads and extracts dataset if not already downloaded. Fetched dataset is
    in the standard .csv format and loaded into an sktime-compatible in-memory
    format (pd_multiindex_hier). For additional information on the dataset,
    including its structure and contents, refer to `Notes` section.


    Examples
    --------
    >>> from sktime.datasets.forecasting import M5Dataset
    >>> dataset = M5Dataset()
    >>> y, X = dataset.load("y", "X")

    Notes
    -----
    The dataset consists of three main files:
    - sales_train_validation.csv: daily sales data for each product and store
    - sell_prices.csv: price data for each product and store
    - calendar.csv: calendar information including events

    The dataframe will have a multi-index with the following levels:
    - state_id
    - store_id
    - dept_id
    - cat_id
    - item_id
    - date

    Dimensionality:     univariate
    Series length:      Approximately 58 million rows (for the full dataset).
    Frequency:          Daily
    Number of features: 8
    Hierarchy levels:   5
    """

    _tags = {
        "name": "m5_forecasting_accuracy",
        "n_splits": 1,
        # Estimator type
        "is_univariate": True,
        "is_one_series": False,
        "is_one_panel": False,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": True,
        "n_instances": 59181090,
        "n_instances_train": 58327370,
        "n_instances_test": 853720,
        "n_timepoints": 1941,
        "n_timepoints_train": 1913,
        "n_timepoints_test": 28,
        "frequency": "D",
        "n_dimensions": 1,
        "n_panels": 30490,  # 30490 bottom levels
        "n_hierarchy_levels": 5,  # Number of levels  in the hierarchy (equivalent to
        #  number of index levels excluding the time index)
    }

    def __init__(
        self,
        extract_path: str = None,
    ):
        self.extract_path = extract_path
        super().__init__()

        self._extract_path = self.cache_files_directory()

    def cache_files_directory(self):
        """Return the path to the directory where the data is stored."""
        if self.extract_path is None:
            return super().cache_files_directory()
        else:
            return Path(self.extract_path)

    @property
    def path_to_data_dir(self):
        """Path to the directory where the data is stored."""
        return self._extract_path / Path("m5-forecasting-accuracy")

    def _download_if_needed(self):
        """Download the data if it is not already downloaded."""
        if not self.path_to_data_dir.exists():
            _download_and_extract(
                "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip",
                extract_path=self._extract_path,
            )

    def _load(self, *args):
        """Load the dataset.

        Downlaods the data if it is not already downloaded, and
        joins the 3 main files into a single dataframe.

        The last 28 days are used as test data, and the rest is used as training data.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            available/valid strings are provided by the concrete classes

        Returns
        -------
        pd.DataFrame
            data container corresponding to string in args (see above)
        """
        self._download_if_needed()

        target_variable = _reduce_memory_usage(
            pd.read_csv(self.path_to_data_dir / Path("sales_train_evaluation.csv"))
        )

        sell_prices = _reduce_memory_usage(
            pd.read_csv(self.path_to_data_dir / Path("sell_prices.csv"))
        )

        calendar = _reduce_memory_usage(
            pd.read_csv(self.path_to_data_dir / Path("calendar.csv"))
        )

        y, X = self._process_raw_data(target_variable, sell_prices, calendar)

        last_28_days = X.index.get_level_values("date").unique()[-28:]
        X_test = X[X.index.get_level_values(-1).isin(last_28_days)]
        y_test = y[y.index.get_level_values(-1).isin(last_28_days)]
        X_train = X[~X.index.get_level_values(-1).isin(last_28_days)]
        y_train = y[~y.index.get_level_values(-1).isin(last_28_days)]

        return self._return_sets(
            args,
            {
                "X": X,
                "y": y,
                "X_test": X_test,
                "y_test": y_test,
                "X_train": X_train,
                "y_train": y_train,
                # "cv": self._load_simple_train_test_cv_split(),
            },
        )

    def _return_sets(self, sets_to_return: list[str], cache: dict[str, pd.DataFrame]):
        """
        Filter the cache dictionary to only include the sets that are requested.

        Parameters
        ----------
        sets_to_return: List[str]
            List of set names to return

        cache: Dict[str, pd.DataFrame]
            Dictionary containing the sets

        Returns
        -------
        Tuple[pd.DataFrame]
            Tuple containing the requested sets
        """
        output = []
        for set_name in sets_to_return:
            output.append(cache[set_name])

        if len(output) == 1:
            return output[0]
        return tuple(output)

    def _process_raw_data(
        self,
        target_variable: pd.DataFrame,
        sell_prices: pd.DataFrame,
        calendar: pd.DataFrame,
    ):
        """
        Process the raw data to create the target variable and the exogenous variables.

        Parameters
        ----------
        target_variable : pd.DataFrame
            Raw data for the target variable
        sell_prices : pd.DataFrame
            Raw data for the sell prices
        calendar : pd.DataFrame
            Raw data for the calendar

        Returns
        -------
        Tuple[pd.DataFrame]
            Tuple containing the processed target variable and the exogenous variables
        """
        df = pd.melt(
            target_variable.drop(columns=["id"]),
            id_vars=[
                "state_id",
                "store_id",
                "cat_id",
                "dept_id",
                "item_id",
            ],
            var_name="day",
            value_name="sales",
        ).dropna()

        # Calendar columns of interest:

        df = df.merge(
            calendar[
                [
                    "d",
                    "event_name_1",
                    "event_type_1",
                    "event_name_2",
                    "event_type_2",
                    "snap_CA",
                    "snap_TX",
                    "snap_WI",
                    "wm_yr_wk",
                ]
            ],
            left_on="day",
            right_on="d",
            how="left",
        )

        df = df.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

        start_date = pd.to_datetime("2011-01-29").to_period("D")
        df["date"] = start_date + df["day"].str.replace("d_", "").astype(int)

        df = df[
            [
                "date",
                "state_id",
                "store_id",
                "cat_id",
                "dept_id",
                "item_id",
                "sales",
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "snap_CA",
                "snap_TX",
                "snap_WI",
                "sell_price",
            ]
        ]
        df = df.set_index(
            ["state_id", "store_id", "dept_id", "cat_id", "item_id", "date"]
        ).sort_index()

        y, X = df[["sales"]], df.drop(columns=["sales"])
        return y, X
