"""Utilities for loading datasets."""

__author__ = [
    "ericjb",
]

__all__ = [
    "load_fpp3",
]

import os
import shutil
import tarfile
import tempfile
import warnings

import pandas as pd

# import zipfile
# from urllib.error import HTTPError, URLError
# from warnings import warn


# from sktime.datasets._data_io import (
#     _download_and_extract,
#     _list_available_datasets,
#     _load_dataset,
#     _load_provided_dataset,
# )
# from sktime.datasets._readers_writers.tsf import load_tsf_to_dataframe
# from sktime.datasets.tsf_dataset_names import tsf_all, tsf_all_datasets


MODULE = os.path.dirname(__file__)


fpp3 = [
    "aus_accommodation",
    "aus_airpassengers",
    "aus_arrivals",
    "bank_calls",
    "boston_marathon",
    "canadian_gas",
    "guinea_rice",
    "insurance",
    "prices",
    "souvenirs",
    "us_change",
    "us_employment",
    "us_gasoline",
]
tsibble = ["pedestrian", "tourism"]

tsibbledata = [
    "ansett",
    "aus_livestock",
    "aus_production",
    "aus_retail",
    "gafa_stock",
    "global_economy",
    "hh_budget",
    "nyc_bikes",
    "olympic_running",
    "PBS",
    "pelt",
    "vic_elec",
]

DATASET_NAMES_FPP3 = fpp3 + tsibble + tsibbledata


def _get_dataset_url(dataset_name):
    url_fpp3 = "https://cran.r-project.org/src/contrib/fpp3_0.5.tar.gz"
    url_tsibble = "https://cran.r-project.org/src/contrib/tsibble_1.1.4.tar.gz"
    url_tsibbledata = "https://cran.r-project.org/src/contrib/tsibbledata_0.4.1.tar.gz"

    if dataset_name in fpp3:
        return (True, url_fpp3)
    if dataset_name in tsibble:
        return (True, url_tsibble)
    if dataset_name in tsibbledata:
        return (True, url_tsibbledata)

    return (False, None)


def _decompress_file_to_temp(url, temp_folder=None):
    import requests

    if temp_folder is None:
        temp_folder = tempfile.gettempdir()
    temp_dir = tempfile.mkdtemp(dir=temp_folder)
    response = requests.get(url)  # noqa: S113
    temp_file = os.path.join(temp_dir, "foo.tar.gz")
    with open(temp_file, "wb") as f:
        f.write(response.content)
    tar = tarfile.open(temp_file)
    tar.extractall(path=temp_dir)
    tar.close()
    return temp_dir


def _find_dataset(temp_folder, dataset_name):
    dataset = dataset_name + ".rda"
    for root, _, files in os.walk(temp_folder):
        if dataset in files:
            return (True, os.path.join(root, dataset))
    return (False, None)


def _yearweek_constructor(obj, attrs):
    return pd.to_datetime(obj, origin="1970-01-01", unit="D").to_period("W").astype(str)


def _yearmonth_constructor(obj, attrs):
    return pd.to_datetime(obj, origin="1970-01-01", unit="D").to_period("M").astype(str)


def _yearquarter_constructor(obj, attrs):
    return pd.to_datetime(obj, origin="1970-01-01", unit="D").to_period("Q").astype(str)


def _date_constructor(obj, attrs):
    return pd.to_datetime(obj, origin="1970-01-01", unit="D")


def _import_rda(path):
    import rdata

    constructor_dict = {
        **rdata.conversion.DEFAULT_CLASS_MAP,
        "Date": _date_constructor,
        "yearweek": _yearweek_constructor,
        "yearmonth": _yearmonth_constructor,
        "yearquarter": _yearquarter_constructor,
    }
    show_warnings = False
    if show_warnings:
        obj = rdata.read_rda(path, constructor_dict=constructor_dict)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = rdata.read_rda(path, constructor_dict=constructor_dict)

    if len(obj) == 1:
        return (True, next(iter(obj.values())))

    return (False, obj)


def _dataset_to_mtype(dataset_name, obj):
    if dataset_name in [
        "aus_airpassengers",
        "guinea_rice",
        "pelt",
        "prices",
        "olympic_running",
        "boston_marathon",
        "global_economy",
        "hh_budget",
    ]:
        if dataset_name in ["prices"]:
            obj.rename(columns={"year": "Year"}, inplace=True)
        obj["Year"] = pd.to_datetime(obj["Year"].astype(int), format="%Y").dt.to_period(
            "Y"
        )
        obj.set_index("Year", inplace=True)
        if dataset_name in ["prices"]:
            obj.index.rename("year", inplace=True)

    if dataset_name == "bank_calls":
        obj["DateTime"] = pd.to_datetime(obj["DateTime"], unit="s")
        obj.set_index("DateTime", inplace=True)

    if dataset_name == "vic_elec":
        obj["Time"] = pd.to_datetime(obj["Time"], unit="s")
        obj.set_index("Time", inplace=True)

    if dataset_name in [
        "canadian_gas",
        "souvenirs",
        "insurance",
        "aus_livestock",
        "us_employment",
        "aus_retail",
        "PBS.csv",
    ]:
        obj["Month"] = pd.to_datetime(obj["Month"], format="%Y-%m").dt.to_period("M")
        obj.set_index("Month", inplace=True)

    if dataset_name in ["us_gasoline", "ansett"]:
        obj.set_index("Week", inplace=True)
        # Extract the start date of each week
        start_dates = obj.index.str.split("/").str[0]
        obj.index = pd.PeriodIndex(start_dates, freq="W-SUN")

    if dataset_name in ["aus_production", "us_change", "tourism"]:
        obj.set_index("Quarter", inplace=True)
        obj.index = pd.PeriodIndex(obj.index, freq="Q")

    if dataset_name in [
        "aus_airpassengers",
        "guinea_rice",
        "bank_calls",
        "canadian_gas",
        "souvenirs",
        "us_gasoline",
    ]:
        obj = obj.squeeze()

    if dataset_name == "aus_arrivals":
        obj.set_index("Quarter", inplace=True)
        obj.index = pd.PeriodIndex(obj.index, freq="Q")
        obj.reset_index(inplace=True)
        obj.set_index(["Origin", "Quarter"], inplace=True)

    if dataset_name == "ansett":
        obj.reset_index(inplace=True)
        obj.set_index(["Airports", "Class", "Week"], inplace=True)

    if dataset_name == "aus_livestock":
        obj.reset_index(inplace=True)
        obj.set_index(["Animal", "State", "Month"], inplace=True)

    if dataset_name == "olympic_running":
        obj.reset_index(inplace=True)
        obj.columns = ["Year", "Length", "Sex", "Time"]
        obj.set_index(["Length", "Sex", "Year"], inplace=True)

    if dataset_name == "tourism":
        obj.reset_index(inplace=True)
        obj.columns = ["Quarter", "Region", "State", "Purpose", "Trips"]
        obj.set_index(["Region", "State", "Purpose", "Quarter"], inplace=True)

    if dataset_name == "aus_accommodation":
        obj.set_index("Date", inplace=True)
        obj.index = pd.PeriodIndex(obj.index, freq="Q")
        obj.reset_index(inplace=True)
        obj.columns = ["Date", "State", "Takings", "Occupancy", "CPI"]
        obj.set_index(["State", "Date"], inplace=True)

    if dataset_name == "boston_marathon":
        obj.reset_index(inplace=True)
        obj.columns = ["Year", "Event", "Champion", "Country", "Time"]
        obj["Time"] = obj["Time"] / 60
        obj.set_index(["Event", "Year"], inplace=True)

    if dataset_name == "gafa_stock":
        obj.reset_index(inplace=True)
        obj.set_index(["Symbol", "Date"], inplace=True)

    if dataset_name == "global_economy":
        obj.reset_index(inplace=True)
        obj.columns = [
            "Year",
            "Country",
            "Code",
            "GDP",
            "Growth",
            "CPI",
            "Imports",
            "Exports",
            "Population",
        ]
        obj.set_index(["Country", "Year"], inplace=True)

    if dataset_name == "hh_budget":
        obj.reset_index(inplace=True)
        obj.columns = [
            "Year",
            "Country",
            "Debt",
            "DI",
            "Expenditure",
            "Savings",
            "Wealth",
            "Unemployment",
        ]
        obj.set_index(["Country", "Year"], inplace=True)

    if dataset_name == "nyc_bikes":
        obj["start_time"] = pd.to_datetime(obj["start_time"], unit="s")
        obj.set_index("start_time", inplace=True)
        obj.reset_index(inplace=True)
        obj.set_index(["bike_id", "start_time"], inplace=True)

    if dataset_name == "pedestrian":
        obj["Date_Time"] = pd.to_datetime(obj["Date_Time"], unit="s")
        obj.set_index("Date_Time", inplace=True)
        obj.reset_index(inplace=True)
        obj["Date"] = pd.to_datetime(obj["Date"])
        obj.set_index(["Sensor", "Date_Time"], inplace=True)

    if dataset_name == "us_employment":
        obj.reset_index(inplace=True)
        obj.set_index(["Series_ID", "Month"], inplace=True)

    if dataset_name == "aus_retail":
        obj.reset_index(inplace=True)
        obj.set_index(["State", "Industry", "Month"], inplace=True)

    if dataset_name == "PBS":
        obj.reset_index(inplace=True)
        obj.set_index(["Concession", "Type", "ATC1", "ATC2", "Month"], inplace=True)

    return (True, obj)


def _process_dataset(dataset_name, temp_folder=None):
    known, url = _get_dataset_url(dataset_name)
    if known:
        temp_dir = _decompress_file_to_temp(url=url, temp_folder=temp_folder)
        found, path = _find_dataset(temp_dir, dataset_name)
        if found:
            ret, obj = _import_rda(path)
        else:
            return (False, None)

        shutil.rmtree(temp_dir)  # cleanup

        if not ret:
            return (False, None)

        do_mtype = True
        if do_mtype:
            result = _dataset_to_mtype(dataset_name, obj)
            return result
        else:
            return (True, obj)
    else:
        return (False, None)


def load_fpp3(dataset, temp_folder=None):
    """Load a dataset from the fpp3 package.

    Returns ``pd.DataFrame`` in one of the valid sktime :term:`mtype` formats,
    depending on the dataset.

    Valid datasets are listed in ``datasets.DATASET_NAMES_FPP3``.

    Requires ``rdata`` and ``requests`` packages in the environment.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load.
        Valid values are listed in ``datasets.DATASET_NAMES_FPP3``.
    temp_folder: str, optional
        Location of temporary data folder for downloading and extracting the dataset.
        Deleted if the operation is successful.

    Returns
    -------
    y : pd.DataFrame
        The loaded data.
        The mtype format is  ``pd.DataFrame`` for single time series,
        ``pd-multiindex`` for collections of time series,
        and ``pd_multiindex_hier`` for hierarchical time series.

    Raises
    ------
    ValueError
        If the dataset is not known.
    RuntimeError
        If there is an error loading the dataset.
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies(["requests", "rdata"])

    if dataset not in DATASET_NAMES_FPP3:
        raise ValueError(
            f"Unknown dataset name in load_fpp3: {dataset}. "
            f"Valid datasets are: {DATASET_NAMES_FPP3}"
        )

    status, y = _process_dataset(dataset, temp_folder)

    if not status:
        raise RuntimeError(f"Error in load_fpp3, dataset = {dataset}.")

    return y
