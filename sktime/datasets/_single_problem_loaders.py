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
    "load_airline",
    "load_plaid",
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
    "load_unit_test",
    "load_uschange",
    "load_UCR_UEA_dataset",
    "load_PBS_dataset",
    "load_gun_point_segmentation",
    "load_electric_devices_segmentation",
    "load_macroeconomic",
    "load_unit_test_tsf",
]

import os
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sktime.datasets._data_io import (
    _load_dataset,
    _load_provided_dataset,
    load_tsf_to_dataframe,
)

DIRNAME = "data"
MODULE = os.path.dirname(__file__)


def load_UCR_UEA_dataset(
    name, split=None, return_X_y=True, return_type=None, extract_path=None
):
    """Load dataset from UCR UEA time series archive.

    Downloads and extracts dataset if not already downloaded. Data is assumed to be
    in the standard .ts format: each row is a (possibly multivariate) time series.
    Each dimension is separated by a colon, each value in a series is comma
    separated. For examples see sktime.datasets.data.tsc. ArrowHead is an example of
    a univariate equal length problem, BasicMotions an equal length multivariate
    problem.

    Parameters
    ----------
    name : str
        Name of data set. If a dataset that is listed in tsc_dataset_names is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from www.timeseriesclassification.com, saving it to
        the extract_path.
    split : None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    return_X_y : bool, optional (default=False)
        it returns two objects, if False, it appends the class labels to the dataframe.
    return_type: None or str, optional (default=None)
        Memory data format specification to return X in, if None then return the default
        "nested_univ" type.
        str can be any other supported Panel mtype,
            for list of mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        commonly used specifications:
            "nested_univ: nested pd.DataFrame, pd.Series in cells
            "numpy3D"/"numpy3d"/"np3D": 3D np.ndarray (instance, variable, time index)
            "numpy2d"/"np2d"/"numpyflat": 2D np.ndarray (instance, time index)
            "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.
    extract_path : str, optional (default=None)
        the path to look for the data. If no path is provided, the function
        looks in `sktime/datasets/data/`.

    Returns
    -------
    X: pd.DataFrame
        The time series data for the problem with n_cases rows and either
        n_dimensions or n_dimensions+1 columns. Columns 1 to n_dimensions are the
        series associated with each case. If return_X_y is False, column
        n_dimensions+1 contains the class labels/target variable.
    y: numpy array, optional
        The class labels for each case in X, returned separately if return_X_y is
        True, or appended to X if False

    Examples
    --------
    >>> from sktime.datasets import load_UCR_UEA_dataset
    >>> X, y = load_UCR_UEA_dataset(name="ArrowHead")
    """
    return _load_dataset(name, split, return_X_y, return_type, extract_path)


def load_plaid(split=None, return_X_y=True):
    """Load the PLAID time series classification problem and returns X and y.

    Example of a univariate problem with unequal length series.

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
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array The class labels for each case in X

    Examples
    --------
    >>> from sktime.datasets import load_plaid
    >>> X, y = load_plaid()
    """
    name = "PLAID"
    return _load_dataset(name, split, return_X_y)


def load_gunpoint(split=None, return_X_y=True):
    """Load the GunPoint time series classification problem and returns X and y.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By default it
        loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Examples
    --------
    >>> from sktime.datasets import load_gunpoint
    >>> X, y = load_gunpoint()

    Notes
    -----
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


def load_osuleaf(split=None, return_X_y=True):
    """Load the OSULeaf time series classification problem and returns X and y.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Examples
    --------
    >>> from sktime.datasets import load_osuleaf
    >>> X, y = load_osuleaf()

    Notes
    -----
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


def load_italy_power_demand(split=None, return_X_y=True):
    """Load ItalyPowerDemand time series classification problem.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Examples
    --------
    >>> from sktime.datasets import load_italy_power_demand
    >>> X, y = load_italy_power_demand()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      24
    Train cases:        67
    Test cases:         1029
    Number of classes:  2

    The data was derived from twelve monthly electrical power demand time series from
    Italy and first used in the paper "Intelligent Icons: Integrating Lite-Weight Data
    Mining and Visualization into GUI Operating Systems". The classification task is to
    distinguish days from Oct to March (inclusive) from April to September.
    Dataset details:
    http://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
    """
    name = "ItalyPowerDemand"
    return _load_dataset(name, split, return_X_y)


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

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> X, y = load_unit_test()

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
    """Load the JapaneseVowels time series classification problem.

    Example of a multivariate problem with unequal length series.

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
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Examples
    --------
    >>> from sktime.datasets import load_japanese_vowels
    >>> X, y = load_japanese_vowels()

    Notes
    -----
    Dimensionality:     multivariate, 12
    Series length:      7-29
    Train cases:        270
    Test cases:         370
    Number of classes:  9

    A UCI Archive dataset. 9 Japanese-male speakers were recorded saying
    the vowels 'a' and 'e'. A '12-degree
    linear prediction analysis' is applied to the raw recordings to
    obtain time-series with 12 dimensions and series lengths between 7 and 29.
    The classification task is to predict the speaker. Therefore,
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


def load_arrow_head(split=None, return_X_y=True, return_type="nested_univ"):
    """
    Load the ArrowHead time series classification problem and returns X and y.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Examples
    --------
    >>> from sktime.datasets import load_arrow_head
    >>> X, y = load_arrow_head()

    Notes
    -----
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
    return _load_provided_dataset(
        name=name, split=split, return_X_y=return_X_y, return_type=return_type
    )


def load_acsf1(split=None, return_X_y=True):
    """Load dataset on power consumption of typical appliances.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Examples
    --------
    >>> from sktime.datasets import load_acsf1
    >>> X, y = load_acsf1()

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

    Dataset details: http://www.timeseriesclassification.com/description.php?Dataset
    =ACSF1
    """
    name = "ACSF1"
    return _load_dataset(name, split, return_X_y)


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
        returns 3D numpy array of shape (n,6,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,6), where each cell is a pd.Series of
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
    """Load the shampoo sales univariate time series dataset for forecasting.

    Returns
    -------
    y : pd.Series/DataFrame
        Shampoo sales dataset

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> y = load_shampoo_sales()

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
    """Load the Longley dataset for forecasting with exogenous variables.

    Parameters
    ----------
    y_name: str, optional (default="TOTEMP")
        Name of target variable (y)

    Returns
    -------
    y: pd.Series
        The target series to be predicted.
    X: pd.DataFrame
        The exogenous time series data for the problem.

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()

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
    """Load the lynx univariate time series dataset for forecasting.

    Returns
    -------
    y : pd.Series/DataFrame
        Lynx sales dataset

    Examples
    --------
    >>> from sktime.datasets import load_lynx
    >>> y = load_lynx()

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
    """Load the airline univariate time series dataset [1].

    Returns
    -------
    y : pd.Series
        Time series

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()

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
    """Load MTS dataset for forecasting Growth rates of personal consumption and income.

    Returns
    -------
    y : pd.Series
        selected column, default consumption
    X : pd.DataFrame
        columns with explanatory variables

    Examples
    --------
    >>> from sktime.datasets import load_uschange
    >>> y, X = load_uschange()

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
    .. [1] Data for "Forecasting: Principles and Practice" (2nd Edition)
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

    Returns
    -------
        X : pd.Series
            Single time series for segmentation
        period_length : int
            The annotated period length by a human expert
        change_points : numpy array
            The change points annotated within the dataset

    Examples
    --------
    >>> from sktime.datasets import load_gun_point_segmentation
    >>> X, period_length, change_points = load_gun_point_segmentation()
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

    Returns
    -------
        X : pd.Series
            Single time series for segmentation
        period_length : int
            The annotated period length by a human expert
        change_points : numpy array
            The change points annotated within the dataset

    Examples
    --------
    >>> from sktime.datasets import load_electric_devices_segmentation
    >>> X, period_length, change_points = load_electric_devices_segmentation()
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
    """Load the Pharmaceutical Benefit Scheme univariate time series dataset [1]_.

    Returns
    -------
    y : pd.Series
     Time series

    Examples
    --------
    >>> from sktime.datasets import load_PBS_dataset
    >>> y = load_PBS_dataset()

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
    .. [1] Data for "Forecasting: Principles and Practice" (3rd Edition)
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
    Load the US Macroeconomic Data [1]_.

    Returns
    -------
    y : pd.DataFrame
     Time series

    Examples
    --------
    >>> from sktime.datasets import load_macroeconomic
    >>> y = load_macroeconomic()

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


def load_solar(
    start="2021-05-01",
    end="2021-09-01",
    normalise=True,
    return_full_df=False,
    api_version="v4",
):
    """Get national solar estimates for GB from Sheffield Solar PV_Live API.

    This function calls the Sheffield Solar PV_Live API to extract national solar data
    for the GB eletricity network. Note that these are estimates of the true solar
    generation, since the true values are "behind the meter" and essentially
    unknown.

    The returned time series is half hourly. For more information please refer
    to [1, 2]_.

    Parameters
    ----------
    start : string, default="2021-05-01"
        The start date of the time-series in "YYYY-MM-DD" format
    end : string, default="2021-09-01"
        The end date of the time-series in "YYYY-MM-DD" format
    normalise : boolean, default=True
        Normalise the returned time-series by installed capacity?
    return_full_df : boolean, default=False
        Return a pd.DataFrame with power, capacity, and normalised estimates?
    api_version : string, default="v4"
        API version to call

    References
    ----------
    .. [1] https://www.solar.sheffield.ac.uk/pvlive/
    .. [2] https://www.solar.sheffield.ac.uk/pvlive/api/

    Examples
    --------
    >>> from sktime.datasets import load_solar  # doctest: +SKIP
    >>> y = load_solar()  # doctest: +SKIP
    """
    from sktime.utils.validation._dependencies import _check_soft_dependencies

    _check_soft_dependencies("backoff")

    import backoff

    @backoff.on_exception(wait_gen=backoff.expo, exception=HTTPError, max_tries=5)
    def _load_solar(
        start="2021-05-01",
        end="2021-09-01",
        normalise=True,
        return_full_df=False,
        api_version="v4",
    ):
        """Private loader, for decoration with backoff."""
        url = "https://api0.solar.sheffield.ac.uk/pvlive/api/"
        url = url + api_version + "/gsp/0?"
        url = url + "start=" + start + "T00:00:00&"
        url = url + "end=" + end + "T00:00:00&"
        url = url + "extra_fields=capacity_mwp&"
        url = url + "data_format=csv"

        df = (
            pd.read_csv(
                url, index_col=["gsp_id", "datetime_gmt"], parse_dates=["datetime_gmt"]
            )
            .droplevel(0)
            .sort_index()
        )
        df = df.asfreq("30T")
        df["generation_pu"] = df["generation_mw"] / df["capacity_mwp"]

        if return_full_df:
            df["generation_pu"] = df["generation_mw"] / df["capacity_mwp"]
            return df
        else:
            if normalise:
                return df["generation_pu"].rename("solar_gen")
            else:
                return df["generation_mw"].rename("solar_gen")

    return _load_solar(
        start=start,
        end=end,
        normalise=normalise,
        return_full_df=return_full_df,
        api_version=api_version,
    )
