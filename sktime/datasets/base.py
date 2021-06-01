# -*- coding: utf-8 -*-
"""Utilities for loading datasets."""
import os
import shutil
import tempfile
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from sktime.utils.data_io import load_from_tsfile_to_dataframe

__all__ = [
    "load_airline",
    "load_arrow_head",
    "load_gunpoint",
    "load_osuleaf",
    "load_italy_power_demand",
    "load_basic_motions",
    "load_japanese_vowels",
    "load_shampoo_sales",
    "load_longley",
    "load_lynx",
    "load_acsf1",
    "load_uschange",
    "load_UCR_UEA_dataset",
    "load_PBS_dataset",
]

__author__ = [
    "Markus Löning",
    "Sajay Ganesh",
    "@big-o",
    "Sebastiaan Koel",
    "Emilia Rose",
]

DIRNAME = "data"
MODULE = os.path.dirname(__file__)


# time series classification data sets
def _download_and_extract(url, extract_path=None):
    """
    Download and unzip datasets (helper function).

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
        if it wasn't succesful

    """
    file_name = os.path.basename(url)
    dl_dir = tempfile.mkdtemp()
    zip_file_name = os.path.join(dl_dir, file_name)
    urlretrieve(url, zip_file_name)

    if extract_path is None:
        extract_path = os.path.join(MODULE, "data/%s/" % file_name.split(".")[0])
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


def _list_downloaded_datasets(extract_path):
    """
    Return a list of all the currently downloaded datasets.

    Modified version of
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L250

    Returns
    -------
    datasets : List
        List of the names of datasets downloaded

    """
    if extract_path is None:
        data_dir = os.path.join(MODULE, DIRNAME)
    else:
        data_dir = extract_path
    datasets = [
        path
        for path in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, path))
    ]
    return datasets


def load_UCR_UEA_dataset(name, split=None, return_X_y=False, extract_path=None):
    """
    Load dataset from UCR UEA time series archive.

    Datasets to be found here: http://www.timeseriesclassification.com/dataset.php
    Downloads and extracts dataset if not already downloaded.

    Parameters
    ----------
    name : str
        Name of data set.
        Possible strings can be found at:
        http://www.timeseriesclassification.com/dataset.php .
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.
    extract_path : str, optional (default=None)
        Default extract path is `sktime/datasets/data/`

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X
    """
    return _load_dataset(name, split, return_X_y, extract_path)


def _load_dataset(name, split, return_X_y, extract_path=None):
    """Load time series classification datasets (helper funciton)."""
    # Allow user to have non standard extract path
    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
        local_dirname = extract_path
    else:
        local_module = MODULE
        local_dirname = DIRNAME

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    if name not in _list_downloaded_datasets(extract_path):
        url = "http://timeseriesclassification.com/Downloads/%s.zip" % name
        # This also tests the validitiy of the URL, can't rely on the html
        # status code as it always returns 200
        try:
            _download_and_extract(url, extract_path)
        except zipfile.BadZipFile as e:
            raise ValueError(
                "Invalid dataset name. Please make sure the dataset is "
                "available on http://timeseriesclassification.com/."
            ) from e

    if split in ("train", "test"):
        fname = name + "_" + split.upper() + ".ts"
        abspath = os.path.join(local_module, local_dirname, name, fname)
        X, y = load_from_tsfile_to_dataframe(abspath)

    # if split is None, load both train and test set
    elif split is None:
        X = pd.DataFrame(dtype="object")
        y = pd.Series(dtype="object")
        for split in ("train", "test"):
            fname = name + "_" + split.upper() + ".ts"
            abspath = os.path.join(local_module, local_dirname, name, fname)
            result = load_from_tsfile_to_dataframe(abspath)
            X = pd.concat([X, pd.DataFrame(result[0])])
            y = pd.concat([y, pd.Series(result[1])])
    else:
        raise ValueError("Invalid `split` value")

    # Return appropriately
    if return_X_y:
        return X, y
    else:
        X["class_val"] = pd.Series(y)
        return X


def load_gunpoint(split=None, return_X_y=False):
    """
    Load the GunPoint time series classification problem and returns X and y.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X
    Details
    -------
    Dimensionality:     univariate
    Series length:      150
    Train cases:        50
    Test cases:         150
    Number of classes:  2
    This dataset involves one female actor and one male actor making a
    motion with their
    hand. The two classes are: Gun-Draw and Point: For Gun-Draw the actors
    have their
    hands by their sides. They draw a replicate gun from a hip-mounted
    holster, point it
    at a target for approximately one second, then return the gun to the
    holster, and
    their hands to their sides. For Point the actors have their gun by their
    sides.
    They point with their index fingers to a target for approximately one
    second, and
    then return their hands to their sides. For both classes, we tracked the
    centroid
    of the actor's right hands in both X- and Y-axes, which appear to be highly
    correlated. The data in the archive is just the X-axis.

    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=GunPoint
    """
    name = "GunPoint"
    return _load_dataset(name, split, return_X_y)


def load_osuleaf(split=None, return_X_y=False):
    """
    Load the OSULeaf time series classification problem and returns X and y.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Details
    -------
    Dimensionality:     univariate
    Series length:      427
    Train cases:        200
    Test cases:         242
    Number of classes:  6

    The OSULeaf data set consist of one dimensional outlines of leaves.
    The series were obtained by color image segmentation and boundary
    extraction (in the anti-clockwise direction) from digitized leaf images
    of six classes: Acer Circinatum, Acer Glabrum, Acer Macrophyllum,
    Acer Negundo, Quercus Garryanaand Quercus Kelloggii for the MSc thesis
    "Content-Based Image Retrieval: Plant Species Identification" by A Grandhi.

    Dataset details: http://www.timeseriesclassification.com/description.php
    ?Dataset=OSULeaf
    """
    name = "OSULeaf"
    return _load_dataset(name, split, return_X_y)


def load_italy_power_demand(split=None, return_X_y=False):
    """
    Load ItalyPowerDemand time series classification problem.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Details
    -------
    Dimensionality:     univariate
    Series length:      24
    Train cases:        67
    Test cases:         1029
    Number of classes:  2

    The data was derived from twelve monthly electrical power demand time
    series from Italy and
    first used in the paper "Intelligent Icons: Integrating Lite-Weight Data
    Mining and
    Visualization into GUI Operating Systems". The classification task is to
    distinguish days
    from Oct to March (inclusive) from April to September.
    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=ItalyPowerDemand
    """
    name = "ItalyPowerDemand"
    return _load_dataset(name, split, return_X_y)


def load_japanese_vowels(split=None, return_X_y=False):
    """
    Load the JapaneseVowels time series classification problem.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
    default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a
        single dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Details
    -------
    Dimensionality:     multivariate, 12
    Series length:      29
    Train cases:        270
    Test cases:         370
    Number of classes:  9

    A UCI Archive dataset. 9 Japanese-male speakers were recorded saying
    the vowels 'a' and 'e'. A '12-degree
    linear prediction analysis' is applied to the raw recordings to
    obtain time-series with 12 dimensions, a
    originally a length between 7 and 29. In this dataset, instances
    have been padded to the longest length,
    29. The classification task is to predict the speaker. Therefore,
    each instance is a transformed utterance,
    12*29 values with a single class label attached, [1...9]. The given
    training set is comprised of 30
    utterances for each speaker, however the test set has a varied
    distribution based on external factors of
    timing and experimental availability, between 24 and 88 instances per
    speaker. Reference: M. Kudo, J. Toyama
    and M. Shimbo. (1999). "Multidimensional Curve Classification Using
    Passing-Through Regions". Pattern
    Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.
    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=JapaneseVowels
    """
    name = "JapaneseVowels"
    return _load_dataset(name, split, return_X_y)


def load_arrow_head(split=None, return_X_y=False):
    """
    Load the ArrowHead time series classification problem and returns X and y.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Details
    -------
    Dimensionality:     univariate
    Series length:      251
    Train cases:        36
    Test cases:         175
    Number of classes:  3

    The arrowhead data consists of outlines of the images of arrowheads. The
    shapes of the
    projectile points are converted into a time series using the angle-based
    method. The
    classification of projectile points is an important topic in
    anthropology. The classes
    are based on shape distinctions such as the presence and location of a
    notch in the
    arrow. The problem in the repository is a length normalised version of
    that used in
    Ye09shapelets. The three classes are called "Avonlea", "Clovis" and "Mix"."

    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=ArrowHead
    """
    name = "ArrowHead"
    return _load_dataset(name, split, return_X_y)


def load_acsf1(split=None, return_X_y=False):
    """
    Load dataset on power consumption of typical appliances.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Details
    -------
    Dimensionality:     univariate
    Series length:      1460
    Train cases:        100
    Test cases:         100
    Number of classes:  10

    The dataset contains the power consumption of typical appliances.
    The recordings are characterized by long idle periods and some high bursts
    of energy consumption when the appliance is active.
    The classes correspond to 10 categories of home appliances;
    mobile phones (via chargers), coffee machines, computer stations
    (including monitor), fridges and freezers, Hi-Fi systems (CD players),
    lamp (CFL), laptops (via chargers), microwave ovens, printers, and
    televisions (LCD or LED)."

    Dataset details: http://www.timeseriesclassification.com/description.php?Dataset
    =ACSF1
    """
    name = "ACSF1"
    return _load_dataset(name, split, return_X_y)


def load_basic_motions(split=None, return_X_y=False):
    """
    Load the  BasicMotions time series classification problem and returns X and y.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Details
    -------
    Dimensionality:     univariate
    Series length:      100
    Train cases:        40
    Test cases:         40
    Number of classes:  4

    The data was generated as part of a student project where four students performed
    four activities whilst wearing a smart watch. The watch collects 3D accelerometer
    and a 3D gyroscope It consists of four classes, which are walking, resting,
    running and badminton. Participants were required to record motion a total of
    five times, and the data is sampled once every tenth of a second, for a ten second
    period.

    Dataset details: http://www.timeseriesclassification.com/description.php?Dataset
    =BasicMotions
    """
    name = "BasicMotions"
    return _load_dataset(name, split, return_X_y)


# forecasting data sets
def load_shampoo_sales():
    """
    Load the shampoo sales univariate time series dataset for forecasting.

    Returns
    -------
    y : pandas Series/DataFrame
        Shampoo sales dataset

    Details
    -------
    This dataset describes the monthly number of sales of shampoo over a 3
    year period.
    The units are a sales count.

    Dimensionality:     univariate
    Series length:      36
    Frequency:          Monthly
    Number of cases:    1


    References
    ----------
    .. [1] Makridakis, Wheelwright and Hyndman (1998) Forecasting: methods
    and applications,
        John Wiley & Sons: New York. Chapter 3.
    """
    name = "ShampooSales"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, squeeze=True, dtype={1: np.float})
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of shampoo sales"
    return y


def load_longley(y_name="TOTEMP"):
    """
    Load the Longley dataset for forecasting with exogenous variables.

    Parameters
    ----------
    y_name: str, optional (default="TOTEMP")
        Name of target variable (y)

    Returns
    -------
    y: pandas.Series
        The target series to be predicted.
    X: pandas.DataFrame
        The exogenous time series data for the problem.

    Details
    -------
    This mulitvariate time series dataset contains various US macroeconomic
    variables from 1947 to 1962 that are known to be highly collinear.

    Dimensionality:     multivariate, 6
    Series length:      16
    Frequency:          Yearly
    Number of cases:    1

    Variable description:

    TOTEMP - Total employment
    GNPDEFL - Gross national product deflator
    GNP - Gross national product
    UNEMP - Number of unemployed
    ARMED - Size of armed forces
    POP - Population

    References
    ----------
    .. [1] Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the
        Electronic Computer from the Point of View of the User."  Journal of
        the American Statistical Association.  62.319, 819-41.
        (https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat)
    """
    name = "Longley"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0)
    data = data.set_index("YEAR")
    data.index = pd.PeriodIndex(data.index, freq="Y", name="Period")
    data = data.astype(np.float)

    # Get target series
    y = data.pop(y_name)
    return y, data


def load_lynx():
    """
    Load the lynx univariate time series dataset for forecasting.

    Returns
    -------
    y : pandas Series/DataFrame
        Lynx sales dataset

    Details
    -------
    The annual numbers of lynx trappings for 1821–1934 in Canada. This
    time-series records the number of skins of
    predators (lynx) that were collected over several years by the Hudson's
    Bay Company. The dataset was
    taken from Brockwell & Davis (1991) and appears to be the series
    considered by Campbell & Walker (1977).

    Dimensionality:     univariate
    Series length:      114
    Frequency:          Yearly
    Number of cases:    1

    Notes
    -----
    This data shows aperiodic, cyclical patterns, as opposed to periodic,
    seasonal patterns.

    References
    ----------
    .. [1] Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988). The New S
    Language. Wadsworth & Brooks/Cole.

    .. [2] Campbell, M. J. and Walker, A. M. (1977). A Survey of statistical
    work on the Mackenzie River series of
    annual Canadian lynx trappings for the years 1821–1934 and a new
    analysis. Journal of the Royal Statistical Society
    series A, 140, 411–431.
    """
    name = "Lynx"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, squeeze=True, dtype={1: np.float})
    y.index = pd.PeriodIndex(y.index, freq="Y", name="Period")
    y.name = "Number of Lynx trappings"
    return y


def load_airline():
    """
    Load the airline univariate time series dataset [1].

    Returns
    -------
    y : pd.Series
     Time series

    Details
    -------
    The classic Box & Jenkins airline data. Monthly totals of international
    airline passengers, 1949 to 1960.

    Dimensionality:     univariate
    Series length:      144
    Frequency:          Monthly
    Number of cases:    1

    Notes
    -----
    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    ..[1] Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series
          Analysis, Forecasting and Control. Third Edition. Holden-Day.
          Series G.
    """
    name = "Airline"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, squeeze=True, dtype={1: np.float})

    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of airline passengers"
    return y


def load_uschange(y_name="Consumption"):
    """
    Load MTS dataset for forecasting Growth rates of personal consumption and income.

    Returns
    -------
    y : pandas Series
        selected column, default consumption
    X : pandas Dataframe
        columns with explanatory variables

    Details
    -------
    Percentage changes in quarterly personal consumption expenditure,
    personal disposable income, production, savings and the
    unemployment rate for the US, 1960 to 2016.


    Dimensionality:     multivariate
    Columns:            ['Quarter', 'Consumption', 'Income', 'Production',
                         'Savings', 'Unemployment']
    Series length:      188
    Frequency:          Quarterly
    Number of cases:    1

    Notes
    -----
    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    ..fpp2: Data for "Forecasting: Principles and Practice" (2nd Edition)
    """
    name = "Uschange"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0, squeeze=True)

    # Sort by Quarter then set simple numeric index
    # TODO add support for period/datetime indexing
    # data.index = pd.PeriodIndex(data.index, freq='Y')
    data = data.sort_values("Quarter")
    data = data.reset_index(drop=True)
    data.index = pd.Int64Index(data.index)
    data.name = name
    y = data[y_name]
    if y_name != "Quarter":
        data = data.drop("Quarter", axis=1)
    X = data.drop(y_name, axis=1)
    return y, X


def load_PBS_dataset():
    """
    Load the Pharmaceutical Benefit Scheme univariate time series dataset [1].

    Returns
    -------
    y : pd.Series
     Time series

    Details
    -------
    The Pharmaceutical Benefits Scheme (PBS) is the Australian government drugs
    subsidy scheme.
    Data comprises of the numbers of scripts sold each month for immune sera
    and immunoglobulin products in Australia.


    Dimensionality:     univariate
    Series length:      204
    Frequency:          Monthly
    Number of cases:    1

    Notes
    -----
    The time series is intermittent, i.e contains small counts,
    with many months registering no sales at all,
    and only small numbers of items sold in other months.

    References
    ----------
    ..fpp3: Data for "Forecasting: Principles and Practice" (3rd Edition)
    """

    name = "PBS_dataset"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, squeeze=True, dtype={1: np.float})

    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of scripts"
    return y
