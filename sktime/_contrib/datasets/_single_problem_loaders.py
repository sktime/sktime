# -*- coding: utf-8 -*-
"""Utilities for loading datasets."""

__author__ = [
    "mloning",
    "sajaysurya",
    "big-o",
    "SebasKoel",
    "Emiliathewolf",
    "TonyBagnall",
    "yairbeer",
    "patrickZIB",
    "aiwalter",
    "jasonlines",
]

__all__ = [
    "load_UCR_UEA_dataset",
    "load_plaid",
    "load_arrow_head",
    "load_gunpoint",
    "load_osuleaf",
    "load_italy_power_demand",
    "load_basic_motions",
    "load_japanese_vowels",
    "load_acsf1",
    "load_unit_test",
    "load_airline",
    "load_shampoo_sales",
    "load_longley",
    "load_lynx",
    "load_uschange",
    "load_PBS_dataset",
    "load_gun_point_segmentation",
    "load_electric_devices_segmentation",
    "load_macroeconomic",
    "load_unit_test_tsf",
]

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sktime._contrib.datasets._data_io import (
    _load_dataset,
    _load_provided_dataset,
    load_tsf_to_dataframe,
)

DIRNAME = "data"
MODULE = os.path.dirname(__file__)


def load_UCR_UEA_dataset(
    name, split=None, return_X_y=True, extract_path=None, return_type=None
):
    """Load dataset from UCR UEA time series archive.

    Downloads and extracts dataset if not already downloaded. Data is assumed to be
    in the standard .ts format: each row is a (possibly multivariate) time series.
    Each dimension is separated by a colon, each value in a series is comma
    separated. For examples see sktime.datasets.data.tsc. ArrowHead is an example of
    a univariate equal length problem, BasicMotions an equal length multivariate
    problem. We assume the problem has m rows, d dimensions of series length m (where m
    can vary between series if not an equal length problem)

    Parameters
    ----------
    name : str
        Name of data set. If a dataset that is listed in tsc_dataset_names is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from www.timeseriesclassification.com, saving it to
        the extract_path.
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By default it
        loads both into a single data structure, otherwise it looks only for files
        of the format <name>_TRAIN.ts or <name>_TEST.ts.
    extract_path: None or str, optional (default=None).
        Location where to look for and/or extract data. If None, it looks in
        datasets/data
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.
    return_type: None or str{"numpy2d", "numpyflat", "numpy3d", "nested_univ"},
    optional (default=None). Controls the returned data structure.

    Returns
    -------
    X:  The time series data for the problem.
        If return_type is either "numpy2d"/"numpyflat", it returns 2D numpy array of
        shape (n,m), if "numpy3d" it returns 3D numpy array of shape (n,1,m) and if
        "nested_univ" or None it returns a nested pandas DataFrame of shape (n,1),
        where each cell is a pd.Series of length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Raises
    ------
    ValueError if an incompatible return_type type is passed: unequal length series
    must be stored in a DataFrame ("nested_univ") and multivariate series calnnot be
    loaded into 2d numpy ("numpy2d"/"numpyflat").
    """
    return _load_dataset(name, split, return_X_y, extract_path, return_type)


def load_plaid(split=None, return_X_y=True):
    """
    PLAID stands for the Plug Load Appliance Identification Dataset.

    Example of a univariate problem with unequal length series. It loads an 11 class
    classification problem with number of cases, n either 1074/537/537 (for split =
    None/Train/Test) of series variable length m.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and 1 column for the time series, where each cell
    is a pd.Series containing the time series.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Notes
    -----
    Dimensionality:     univariate
    Series length:      Variable
    Train cases:        537
    Test cases:         537
    Number of classes:  11

    """
    name = "PLAID"
    return _load_provided_dataset(name, split, return_X_y)


def load_gunpoint(split=None, return_X_y=True, return_type=None):
    """
    Load data of two actors making a motion with their hand.

    This is an equal length univariate time series classification problem. It loads
    a two class classification problem with number of cases, n either 200/50/150 (for
    split =None/Train/Test) of series length m=150.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By default it
        loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a concatenated data
        structure.
    return_type: None or str{"numpy2d", "numpyflat", "numpy3d", "nested_univ"},
    optional (default=None). Controls the returned data structure.

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Notes
    -----
    Dimensionality:     univariate
    Series length:      150
    Train cases:        50
    Test cases:         150
    Number of classes:  2

    This dataset involves one female actor and one male actor making a motion with their
    hand. The two classes are: Gun-Draw and Point: For Gun-Draw the actors have their
    hands by their sides. They draw a replicate gun from a hip-mounted holster, point it
    at a target for approximately one second, then return the gun to the holster, and
    their hands to their sides. For Point the actors have their gun by their sides.
    They point with their index fingers to a target for approximately one second, and
    then return their hands to their sides. For both classes, they tracked the centroid
    of the actor's right hands in both X- and Y-axes, which appear to be highly
    correlated. The data in the archive is just the X-axis.

    Dataset details:
    http://timeseriesclassification.com/description.php?Dataset=GunPoint
    """
    name = "GunPoint"
    return _load_provided_dataset(name, split, return_X_y, return_type)


def load_osuleaf(split=None, return_X_y=True, return_type=None):
    """
    Load OSULeaf data set, which consists of one dimensional outlines of leaves.

    This is an equal length univariate time series classification problem. It loads
    a six class classification problem with number of cases, n either 442/200/242 (for
    split =None/Train/Test) of series length m=427.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a concatenated data
        structure.
    return_type: None or str{"numpy2d", "numpyflat", "numpy3d", "nested_univ"},
    optional (default=None). Controls the returned data structure.

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Notes
    -----
    Dimensionality:     univariate
    Series length:      427
    Train cases:        200
    Test cases:         242
    Number of classes:  6

    The OSULeaf data set consist of one dimensional outlines of leaves. The series
    were obtained by color image segmentation and boundary extraction (in the
    anti-clockwise direction) from digitized leaf images of six classes: Acer
    Circinatum, Acer Glabrum, Acer Macrophyllum, Acer Negundo, Quercus Garryanaand
    Quercus Kelloggii for the MSc thesis "Content-Based Image Retrieval: Plant
    Species Identification" by A Grandhi.

    Dataset details:
    http://www.timeseriesclassification.com/description.php?Dataset=OSULeaf
    """
    name = "OSULeaf"
    return _load_provided_dataset(name, split, return_X_y, return_type)


def load_italy_power_demand(split=None, return_X_y=True, return_type=None):
    """
    Load 12 monthly electrical power demand time series from Italy.

    This is an equal length univariate time series classification problem. It loads
    a two class classification problem with number of cases, n either 1096/67/1029 (for
    split =None/Train/Test) of series length m=24.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a concatenated data
        structure.
    return_type: None or str{"numpy2d", "numpyflat", "numpy3d", "nested_univ"},
    optional (default=None). Controls the returned data structure.

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Notes
    -----
     Dimensionality:     univariate
     Series length (m):      24
     Train cases:        67
     Test cases:         1029
     Number of classes:  2

    The data was first used in the paper "Intelligent Icons: Integrating Lite-Weight
    Data Mining and Visualization into GUI Operating Systems". The classification
    task is to distinguish days from Oct to March (inclusive) from April to September.
    Dataset details:
    http://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
    """
    name = "ItalyPowerDemand"
    return _load_provided_dataset(name, split, return_X_y, return_type)


def load_unit_test(split=None, return_X_y=True, return_type=None):
    """
    Load UnitTest data.

    This is an equal length univariate time series classification problem. It is a
    stripped down version of the ChinaTown problem that is used in correctness tests
    for classification. It loads a two class classification problem with number of
    cases, n, where n = 42 (if split is None) or 20/22 (if split is "train"/"test")
    of series length m = 24

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By default it
        loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a concatenated data
        structure.
    return_type: None or str{"numpy2d", "numpyflat", "numpy3d", "nested_univ"},
    optional (default=None). Controls the returned data structure.

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Details
    -------
    This is the Chinatown problem with a smaller test set, useful for rapid tests.
    Dimensionality:     univariate
    Series length:      24
    Train cases:        20
    Test cases:         22 (full dataset has 345)
    Number of classes:  2

     See
    http://timeseriesclassification.com/description.php?Dataset=Chinatown
    for the full dataset
    """
    name = "UnitTest"
    return _load_provided_dataset(name, split, return_X_y, return_type)


def load_japanese_vowels(split=None, return_X_y=True):
    """
    Load the japanese vowels audio problem.

    This is a multivariate, unequal length time series classification problem. It
    loads a nine class classification problem with number of cases, n, where n = 640
    (if split is None) or 270/370 (if split is "train"/"test") of series variable
    length m.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
    default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a
        single dataframe with columns for features and the target.

    Returns
    -------
    X: pandas DataFrame with n rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Notes
    -----
    Dimensionality:     multivariate, 12
    Series length:      7-29
    Train cases:        270
    Test cases:         370
    Number of classes:  9

    A UCI Archive dataset. 9 Japanese-male speakers were recorded saying the vowels
    'a' and 'e'. A '12-degree linear prediction analysis' is applied to the raw
    recordings to obtain time-series with 12 dimensions and series lengths between 7 and
    29. The classification task is to predict the speaker. Therefore, each instance
    is a transformed utterance, 12*29 values with a single class label attached,
    [1...9]. The given training set is comprised of 30 utterances for each speaker,
    however the test set has a varied distribution based on external factors of
    timing and experimental availability, between 24 and 88 instances per
    speaker. Reference: M. Kudo, J. Toyama and M. Shimbo. (1999). "Multidimensional
    Curve Classification Using Passing-Through Regions". Pattern Recognition Letters,
    Vol. 20, No. 11--13, pages 1103--1111. Dataset details:
    http://timeseriesclassification.com/description.php?Dataset=JapaneseVowels
    """
    name = "JapaneseVowels"
    return _load_provided_dataset(name, split, return_X_y)


def load_arrow_head(split=None, return_X_y=True, return_type=None):
    """
    Load dataset of arrow head shape outlines.

    This is an equal length univariate time series classification problem. It loads a
    three class classification problem with number of cases, n, where n = 211 (if
    split is None) or 36/175 (if split is "train"/"test") of series length m = 251

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (time series, target) separately as X and y instead of a single
        data structure.
    return_type: None or str{"numpy2d", "numpyflat", "numpy3d", "nested_univ"},
    optional (default=None). Controls the returned data structure.

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Notes
    -----
    Dimensionality:     univariate
    Series length:      251
    Train cases:        36
    Test cases:         175
    Number of classes:  3

    The arrowhead data consists of outlines of the images of arrowheads. The
    shapes of the projectile points are converted into a time series using the angle
    based method. The classification of projectile points is an important topic in
    anthropology. The classes are based on shape distinctions such as the presence and
    location of a notch in the arrow. The problem in the repository is a length
    normalised version of that used in [1]. The three classes are called "Avonlea",
    "Clovis" and "Mix".

    Dataset:
    http://timeseriesclassification.com/description.php?Dataset=ArrowHead
    """
    name = "ArrowHead"
    return _load_provided_dataset(
        name=name, split=split, return_X_y=return_X_y, return_type=return_type
    )


def load_acsf1(split=None, return_X_y=True, return_type=None):
    """Load dataset on power consumption of typical appliances.

    This is an equal length univariate time series classification problem. It loads a
    10 class classification problem with number of cases, n, where n = 200 (if
    split is None) or 100 (if split is "train"/"test") of series length m = 1460

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (time series, target) separately as X and y instead of a single
        data structure.
    return_type: None or str{"numpy2d", "numpyflat", "numpy3d", "nested_univ"},
    optional (default=None). Controls the returned data structure.

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Notes
    -----
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

    Dataset details:
    http://www.timeseriesclassification.com/description.php?Dataset=ACSF1
    """
    name = "ACSF1"
    return _load_provided_dataset(
        name=name, split=split, return_X_y=return_X_y, return_type=return_type
    )


def load_basic_motions(split=None, return_X_y=True, return_type=None):
    """
    Load the BasicMotions time series classification problem and returns X and y.

    This is an equal length multivariate time series classification problem. It loads a
    4 class classification problem with number of cases, n, where n = 80 (if
    split is None) or 40 (if split is "train"/"test") of series length m = 100.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (time series, target) separately as X and y instead of a single
        data structure.
    return_type: None or str{"numpy3d", "nested_univ"},
    optional (default=None). Controls the returned data structure.

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Raises
    ------
    ValueError if argument "numpy2d"/"numpyflat" is passed as return_type
    Notes
    -----
    Dimensionality:     multivariate, 6
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
    if return_type == "numpy2d" or return_type == "numpyflat":
        raise ValueError(
            f"{name} loader: Error, attempting to load into a numpy2d "
            f"array, but cannot because it is a multivariate problem. Use "
            f"numpy3d instead"
        )
    return _load_provided_dataset(
        name=name, split=split, return_X_y=return_X_y, return_type=return_type
    )


# forecasting data sets
def load_shampoo_sales():
    """
    Load the shampoo sales univariate time series dataset for forecasting.

    Returns
    -------
    y : pandas Series/DataFrame
        Shampoo sales dataset

    Notes
    -----
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
    y = pd.read_csv(path, index_col=0, dtype={1: float}).squeeze("columns")
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

    Notes
    -----
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
    data = data.astype(float)

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

    Notes
    -----
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
    y = pd.read_csv(path, index_col=0, dtype={1: float}).squeeze("columns")
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

    Notes
    -----
    The classic Box & Jenkins airline data. Monthly totals of international
    airline passengers, 1949 to 1960.

    Dimensionality:     univariate
    Series length:      144
    Frequency:          Monthly
    Number of cases:    1

    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    .. [1] Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series
          Analysis, Forecasting and Control. Third Edition. Holden-Day.
          Series G.
    """
    name = "Airline"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, dtype={1: float}).squeeze("columns")

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

    Notes
    -----
    Percentage changes in quarterly personal consumption expenditure,
    personal disposable income, production, savings and the
    unemployment rate for the US, 1960 to 2016.


    Dimensionality:     multivariate
    Columns:            ['Quarter', 'Consumption', 'Income', 'Production',
                         'Savings', 'Unemployment']
    Series length:      188
    Frequency:          Quarterly
    Number of cases:    1

    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    ..fpp2: Data for "Forecasting: Principles and Practice" (2nd Edition)
    """
    name = "Uschange"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0).squeeze("columns")

    # Sort by Quarter then set simple numeric index
    # TODO add support for period/datetime indexing
    # data.index = pd.PeriodIndex(data.index, freq='Y')
    data = data.sort_values("Quarter")
    data = data.reset_index(drop=True)
    data.index = pd.Index(data.index, dtype=int)
    data.name = name
    y = data[y_name]
    if y_name != "Quarter":
        data = data.drop("Quarter", axis=1)
    X = data.drop(y_name, axis=1)
    return y, X


def load_gun_point_segmentation():
    """Load the GunPoint time series segmentation problem and returns X.

    We group TS of the UCR GunPoint dataset by class label and concatenate
    all TS to create segments with repeating temporal patterns and
    characteristics. The location at which different classes were
    concatenated are marked as change points.

    We resample the resulting TS to control the TS resolution.
    The window sizes for these datasets are hand-selected to capture
    temporal patterns but are approximate and limited to the values
    [10,20,50,100] to avoid over-fitting.

    -----------

    Returns
    -------
        X : pd.Series
            Single time series for segmentation
        period_length : int
            The annotated period length by a human expert
        change_points : numpy array
            The change points annotated within the dataset
    -----------
    """
    dir = "segmentation"
    name = "GunPoint"
    fname = name + ".csv"

    period_length = int(10)
    change_points = np.int32([900])

    path = os.path.join(MODULE, DIRNAME, dir, fname)
    ts = pd.read_csv(path, index_col=0, header=None).squeeze("columns")

    return ts, period_length, change_points


def load_electric_devices_segmentation():
    """Load the Electric Devices segmentation problem and returns X.

    We group TS of the UCR Electric Devices dataset by class label and concatenate
    all TS to create segments with repeating temporal patterns and
    characteristics. The location at which different classes were
    concatenated are marked as change points.

    We resample the resulting TS to control the TS resolution.
    The window sizes for these datasets are hand-selected to capture
    temporal patterns but are approximate and limited to the values
    [10,20,50,100] to avoid over-fitting.

    -----------

    Returns
    -------
        X : pd.Series
            Single time series for segmentation
        period_length : int
            The annotated period length by a human expert
        change_points : numpy array
            The change points annotated within the dataset
    -----------
    """
    dir = "segmentation"
    name = "ElectricDevices"
    fname = name + ".csv"

    period_length = int(10)
    change_points = np.int32([1090, 4436, 5712, 7923])

    path = os.path.join(MODULE, DIRNAME, dir, fname)
    ts = pd.read_csv(path, index_col=0, header=None).squeeze("columns")

    return ts, period_length, change_points


def load_PBS_dataset():
    """Load the Pharmaceutical Benefit Scheme univariate time series dataset [1].

    Returns
    -------
    y : pd.Series
     Time series

    Notes
    -----
    The Pharmaceutical Benefits Scheme (PBS) is the Australian government drugs
    subsidy scheme.
    Data comprises of the numbers of scripts sold each month for immune sera
    and immunoglobulin products in Australia.


    Dimensionality:     univariate
    Series length:      204
    Frequency:          Monthly
    Number of cases:    1

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
    y = pd.read_csv(path, index_col=0, dtype={1: float}).squeeze("columns")

    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of scripts"
    return y


def load_macroeconomic():
    """
    Load the US Macroeconomic Data [1].

    Returns
    -------
    y : pd.DataFrame
     Time series

    Notes
    -----
    US Macroeconomic Data for 1959Q1 - 2009Q3.

    Dimensionality:     multivariate, 14
    Series length:      203
    Frequency:          Quarterly
    Number of cases:    1

    This data is kindly wrapped via `statsmodels.datasets.macrodata`.

    References
    ----------
    .. [1] Wrapped via statsmodels:
          https://www.statsmodels.org/dev/datasets/generated/macrodata.html
    .. [2] Data Source: FRED, Federal Reserve Economic Data, Federal Reserve
          Bank of St. Louis; http://research.stlouisfed.org/fred2/;
          accessed December 15, 2009.
    .. [3] Data Source: Bureau of Labor Statistics, U.S. Department of Labor;
          http://www.bls.gov/data/; accessed December 15, 2009.
    """
    y = sm.datasets.macrodata.load_pandas().data
    y["year"] = y["year"].astype(int).astype(str)
    y["quarter"] = y["quarter"].astype(int).astype(str).apply(lambda x: "Q" + x)
    y["time"] = y["year"] + "-" + y["quarter"]
    y.index = pd.PeriodIndex(data=y["time"], freq="Q", name="Period")
    y = y.drop(columns=["year", "quarter", "time"])
    y.name = "US Macroeconomic Data"
    return y


def load_unit_test_tsf():
    """
    Load tsf UnitTest dataset.

    Returns
    -------
    loaded_data: pd.DataFrame
        The converted dataframe containing the time series.
    frequency: str
        The frequency of the dataset.
    forecast_horizon: int
        The expected forecast horizon of the dataset.
    contain_missing_values: bool
        Whether the dataset contains missing values or not.
    contain_equal_length: bool
        Whether the series have equal lengths or not.
    """
    path = os.path.join(MODULE, DIRNAME, "UnitTest", "UnitTest_Tsf_Loader.tsf")
    (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = load_tsf_to_dataframe(path)

    return (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    )
