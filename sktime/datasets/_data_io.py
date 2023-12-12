"""Utilities for loading panel datasets."""

__author__ = ["Emiliathewolf", "TonyBagnall", "jasonlines", "achieveordie"]

__all__ = [
    "generate_example_long_table",
    "make_multi_index_dataframe",
    "_load_dataset",
]

import os
import shutil
import tempfile
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from sktime.datasets._readers_writers.ts import load_from_tsfile
from sktime.datasets._readers_writers.utils import _alias_mtype_check
from sktime.datatypes import convert
from sktime.datatypes._panel._convert import _make_column_names

DIRNAME = "data"
MODULE = os.path.dirname(__file__)
CLASSIF_URLS = [
    "https://timeseriesclassification.com/aeon-toolkit",  # main mirror (UEA)
    "https://github.com/sktime/sktime-datasets/raw/main/TSC",  # backup mirror (sktime)
]


def _download_and_extract(url, extract_path=None):
    """Download and unzip datasets (helper function).

    This code was modified from
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L28

    Parameters
    ----------
    url : string
        Url pointing to file to download
    extract_path : string, optional (default: None)
        path to extract downloaded zip to, None defaults
        to sktime/datasets/data

    Returns
    -------
    extract_path : string or None
        if successful, string containing the path of the extracted file, None
        if it wasn't successful
    """
    file_name = os.path.basename(url)
    dl_dir = tempfile.mkdtemp()
    zip_file_name = os.path.join(dl_dir, file_name)
    urlretrieve(url, zip_file_name)

    if extract_path is None:
        extract_path = os.path.join(MODULE, "local_data/%s/" % file_name.split(".")[0])
    else:
        extract_path = os.path.join(extract_path, "%s/" % file_name.split(".")[0])

    try:
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        zipfile.ZipFile(zip_file_name, "r").extractall(extract_path)
        shutil.rmtree(dl_dir)
        return extract_path
    except zipfile.BadZipFile:
        shutil.rmtree(dl_dir)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        raise zipfile.BadZipFile(
            "Could not unzip dataset. Please make sure the URL is valid."
        )


def _list_available_datasets(extract_path, origin_repo=None):
    """Return a list of all the currently downloaded datasets.

    Forecastingorg datasets are in the format <dataset_name>.tsf while classification
    are in the format <dataset_name>_TRAIN.ts and <dataset_name>_TEST.ts.
    To count as available, each directory <dir_name>
    in the extract_path must contain files called
    1. <dir_name>_TRAIN.ts and <dir_name>_TEST.ts if datasets from classification repo.
    2. <dir_name>.tsf if datasets from forecasting repo.

    Parameters
    ----------
    extract_path: string
        root directory where to look for files, if None defaults to sktime/datasets/data
    origin_repo: string, optional (default=None)
        if None, returns all available classification datasets in extract_path,
        if string (must be "forecastingorg"), returns all available
        forecastingorg datasets in extract_path.

    Returns
    -------
    datasets : List
        List of the names of datasets downloaded
    """
    if extract_path is None:
        data_dir = os.path.join(MODULE, "data")
    else:
        data_dir = extract_path
    datasets = []
    for name in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, name)
        if os.path.isdir(sub_dir):
            all_files = os.listdir(sub_dir)
            if origin_repo == "forecastingorg":
                if name + ".tsf" in all_files:
                    datasets.append(name)
            else:
                if name + "_TRAIN.ts" in all_files and name + "_TEST.ts" in all_files:
                    datasets.append(name)
    return datasets


def _cache_dataset(url, name, extract_path=None, repeats=1, verbose=False):
    """Download and unzip datasets from multiple mirrors or fallback sources.

    If url is string, will attempt to download and unzip from url, to extract_path.
    If url is list of str, will go through urls in order until a download succeeds.

    Parameters
    ----------
    url : string or list of string
        URL pointing to file to download
        files are expected to be at f"{url}/{name}.zip" for a string url
        or f"{url[i]}/{name}.zip" for a list of string urls
    extract_path : string, optional (default: None)
        path to extract downloaded zip to, None defaults
        to sktime/datasets/data
    repeats : int, optional (default: 1)
        number of times to try downloading from each url
    verbose : bool, optional (default: False)
        whether to print progress

    Returns
    -------
    extract_path : string or None
        if successful, string containing the path of the extracted file
    u : string
        url from which the dataset was downloaded
    repeat : int
        number of times it took to download the dataset from u
    If none of the attempts are successful, will raise RuntimeError
    """
    if isinstance(url, str):
        url = [url]

    for u in url:
        name_url = f"{u}/{name}.zip"
        for repeat in range(repeats):
            if verbose:
                print(  # noqa: T201
                    f"Downloading dataset {name} from {u} to {extract_path}"
                    f"(attempt {repeat} of {repeats} total). "
                )

            try:
                _download_and_extract(name_url, extract_path=extract_path)
                return extract_path, u, repeat

            except zipfile.BadZipFile:
                if verbose:
                    if repeat < repeats - 1:
                        print(  # noqa: T201
                            "Download failed, continuing with next attempt. "
                        )
                    else:
                        print(  # noqa: T201
                            "All attempts for mirror failed, "
                            "continuing with next mirror."
                        )

    raise RuntimeError(
        f"Dataset with name ={name} could not be downloaded from any of the mirrors."
    )


def _mkdir_if_not_exist(*path):
    """Shortcut for making a directory if it does not exist.

    Parameters
    ----------
    path : tuple of strings
        Directory path to create
        If multiple strings are given, they will be joined together

    Returns
    -------
    os.path.join(*path) : string
        Directory path created
    """
    full_path = os.path.join(*path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path


def _load_dataset(name, split, return_X_y, return_type=None, extract_path=None):
    """Load time series classification datasets (helper function).

    Parameters
    ----------
    name : string, file name to load from
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.
    return_type: valid Panel mtype str or None, optional (default=None="nested_univ")
        Memory data format specification to return X in, None = "nested_univ" type.
        str can be any supported sktime Panel mtype,
            for list of mtypes, see datatypes.MTYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        commonly used specifications:
            "nested_univ: nested pd.DataFrame, pd.Series in cells
            "numpy3D"/"numpy3d"/"np3D": 3D np.ndarray (instance, variable, time index)
            "numpy2d"/"np2d"/"numpyflat": 2D np.ndarray (instance, time index)
            "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.
    extract_path : string, optional (default: None)
        path to extract downloaded zip to
        None defaults to sktime/datasets/data if the data exists there, otherwise
        defaults to sktime/datasets/local_data and downloads data there

    Returns
    -------
    X: sktime data container, following mtype specification `return_type`
        The time series data for the problem, with n instances
    y: 1D numpy array of length n, only returned if return_X_y if True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.
    """
    # Allow user to have non standard extract path
    if extract_path is None:
        # default for first check is sktime/datasets/data
        check_path = os.path.join(MODULE, "data")

    def _get_data_from(path):
        return _load_provided_dataset(name, split, return_X_y, return_type, path)

    # if the dataset exists in check_path = sktime/datasets/data, retrieve it from there
    if name in _list_available_datasets(check_path):
        return _get_data_from(check_path)

    # now we know the dataset is not in check_path
    # so we need to check whether it is already in the download/cache path
    # download path is extract_path/local_data, defaults to sktime/datasets/local_data
    if extract_path is None:
        extract_path = os.path.join(MODULE, "local_data")

    # in either case below, we need to ensure the directory exists
    _mkdir_if_not_exist(extract_path)

    # search if the dataset is already in the extract path after download
    if name in _list_available_datasets(extract_path):
        return _get_data_from(extract_path)

    # now we know the dataset is not in the download/cache path
    # so we need to download it

    # download the dataset from CLASSIF_URLS
    # will try multiple mirrors if necessary
    # if fails, will raise a RuntimeError
    _cache_dataset(CLASSIF_URLS, name, extract_path=extract_path)

    # if we reach this, the data has been downloaded, now we can load it
    return _get_data_from(extract_path)


def _load_provided_dataset(
    name,
    split=None,
    return_X_y=True,
    return_type=None,
    extract_path=None,
):
    """Load baked in time series classification datasets (helper function).

    Loads data from the provided files from sktime/datasets/data only.

    Parameters
    ----------
    name : string, file name to load from
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.
    return_type: valid Panel mtype str or None, optional (default=None="nested_univ")
        Memory data format specification to return X in, None = "nested_univ" type.
        str can be any supported sktime Panel mtype,
            for list of mtypes, see datatypes.MTYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        commonly used specifications:
            "nested_univ: nested pd.DataFrame, pd.Series in cells
            "numpy3D"/"numpy3d"/"np3D": 3D np.ndarray (instance, variable, time index)
            "numpy2d"/"np2d"/"numpyflat": 2D np.ndarray (instance, time index)
            "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.
    extract_path: default = join(MODULE, DIRNAME) = os.path.dirname(__file__) + "/data"
        path to extract downloaded zip to

    Returns
    -------
    X: sktime data container, following mtype specification `return_type`
        The time series data for the problem, with n instances
    y: 1D numpy array of length n, only returned if return_X_y if True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.
    """
    if extract_path is None:
        extract_path = os.path.join(MODULE, DIRNAME)

    if isinstance(split, str):
        split = split.upper()

    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        abspath = os.path.join(extract_path, name, fname)
        X, y = load_from_tsfile(abspath, return_data_type="nested_univ")
    # if split is None, load both train and test set
    elif split is None:
        fname = name + "_TRAIN.ts"
        abspath = os.path.join(extract_path, name, fname)
        X_train, y_train = load_from_tsfile(abspath, return_data_type="nested_univ")

        fname = name + "_TEST.ts"
        abspath = os.path.join(extract_path, name, fname)
        X_test, y_test = load_from_tsfile(abspath, return_data_type="nested_univ")

        X = pd.concat([X_train, X_test])
        X = X.reset_index(drop=True)
        y = np.concatenate([y_train, y_test])

    else:
        raise ValueError("Invalid `split` value =", split)

    return_type = _alias_mtype_check(return_type)
    if return_X_y:
        X = convert(X, from_type="nested_univ", to_type=return_type)
        return X, y
    else:
        X["class_val"] = pd.Series(y)
        X = convert(X, from_type="nested_univ", to_type=return_type)
        return X


# left here for now, better elsewhere later perhaps
def generate_example_long_table(num_cases=50, series_len=20, num_dims=2):
    """Generate example from long table format file.

    Parameters
    ----------
    num_cases: int
        Number of cases.
    series_len: int
        Length of the series.
    num_dims: int
        Number of dimensions.

    Returns
    -------
    DataFrame
    """
    rows_per_case = series_len * num_dims
    total_rows = num_cases * series_len * num_dims

    case_ids = np.empty(total_rows, dtype=int)
    idxs = np.empty(total_rows, dtype=int)
    dims = np.empty(total_rows, dtype=int)
    vals = np.random.rand(total_rows)

    for i in range(total_rows):
        case_ids[i] = int(i / rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem / series_len)
        idxs[i] = rem % series_len

    df = pd.DataFrame()
    df["case_id"] = pd.Series(case_ids)
    df["dim_id"] = pd.Series(dims)
    df["reading_id"] = pd.Series(idxs)
    df["value"] = pd.Series(vals)
    return df


def make_multi_index_dataframe(n_instances=50, n_columns=3, n_timepoints=20):
    """Generate example multi-index DataFrame.

    Parameters
    ----------
    n_instances : int
        Number of instances.
    n_columns : int
        Number of columns (series) in multi-indexed DataFrame.
    n_timepoints : int
        Number of timepoints per instance-column pair.

    Returns
    -------
    mi_df : pd.DataFrame
        The multi-indexed DataFrame with
        shape (n_instances*n_timepoints, n_column).
    """
    # Make long DataFrame
    long_df = generate_example_long_table(
        num_cases=n_instances, series_len=n_timepoints, num_dims=n_columns
    )
    # Make Multi index DataFrame
    mi_df = long_df.set_index(["case_id", "reading_id"]).pivot(columns="dim_id")
    mi_df.columns = _make_column_names(n_columns)
    return mi_df
