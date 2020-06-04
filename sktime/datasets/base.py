"""
Utilities for loading datasets
"""

import os

import pandas as pd

from ..utils.load_data import load_from_tsfile_to_dataframe

__all__ = [
    "load_airline",
    "load_gunpoint",
    "load_arrow_head",
    "load_italy_power_demand",
    "load_basic_motions",
    "load_japanese_vowels",
    "load_shampoo_sales",
    "load_longley",
    "load_lynx"
]

__author__ = ['Markus Löning', 'Sajay Ganesh', '@big-o']

DIRNAME = 'data'
MODULE = os.path.dirname(__file__)


# time series classification data sets
def _load_dataset(name, split, return_X_y):
    """
    Helper function to load time series classification datasets.
    """

    if split in ("train", "test"):
        fname = name + '_' + split.upper() + '.ts'
        abspath = os.path.join(MODULE, DIRNAME, name, fname)
        X, y = load_from_tsfile_to_dataframe(abspath)

    # if split is None, load both train and test set
    elif split is None:
        X = pd.DataFrame(dtype="object")
        y = pd.Series(dtype="object")
        for split in ("train", "test"):
            fname = name + '_' + split.upper() + '.ts'
            abspath = os.path.join(MODULE, DIRNAME, name, fname)
            result = load_from_tsfile_to_dataframe(abspath)
            X = pd.concat([X, pd.DataFrame(result[0])])
            y = pd.concat([y, pd.Series(result[1])])
    else:
        raise ValueError("Invalid `split` value")

    # Return appropriately
    if return_X_y:
        return X, y
    else:
        X['class_val'] = pd.Series(y)
        return X


def load_gunpoint(split=None, return_X_y=False):
    """
    Loads the GunPoint time series classification problem and returns X and y

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
    name = 'GunPoint'
    return _load_dataset(name, split, return_X_y)


def load_italy_power_demand(split=None, return_X_y=False):
    """
    Loads the ItalyPowerDemand time series classification problem and
    returns X and y

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

    name = 'ItalyPowerDemand'
    return _load_dataset(name, split, return_X_y)


def load_japanese_vowels(split=None, return_X_y=False):
    """
        Loads the JapaneseVowels time series classification problem and
        returns X and y.

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
        timing and experimenal availability, between 24 and 88 instances per
        speaker. Reference: M. Kudo, J. Toyama
        and M. Shimbo. (1999). "Multidimensional Curve Classification Using
        Passing-Through Regions". Pattern
        Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.

        Dataset details: http://timeseriesclassification.com/description.php
        ?Dataset=JapaneseVowels
    """

    name = 'JapaneseVowels'
    return _load_dataset(name, split, return_X_y)


def load_arrow_head(split=None, return_X_y=False):
    """
    Loads the ArrowHead time series classification problem and returns X and y.

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

    name = 'ArrowHead'
    return _load_dataset(name, split, return_X_y)


def load_basic_motions(split=None, return_X_y=False):
    """
    Loads the ArrowHead time series classification problem and returns X and y.

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

    name = 'BasicMotions'
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

    name = 'ShampooSales'
    fname = name + '.csv'
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0, squeeze=True)

    # change period index to simple numeric index
    # TODO add support for period/datetime indexing
    # data.index = pd.PeriodIndex(data.index, freq='M')
    data = data.reset_index(drop=True)
    data.index = pd.Int64Index(data.index)
    data.name = name
    return data


def load_longley(return_X_y=False):
    """
    Load the Longley multivariate time series dataset for forecasting with
    exogenous variables.

    Parameters
    ----------
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas.DataFrame
        The exogenous time series data for the problem.
    y: pandas.Series
        The target series to be predicted.

    Details
    -------
    This dataset contains various US macroeconomic variables from 1947 to
    1962 that are known to be highly
    collinear.

    Dimensionality:     multivariate, 6
    Series length:      16
    Frequency:          Yearly
    Number of cases:    1

    Variable description:

    TOTEMP - Total employment (y)
    GNPDEFL - Gross national product deflator
    GNP - Gross national product
    UNEMP - Number of unemployed
    ARMED - Size of armed forces
    POP - Population
    YEAR - Calendar year (index)

    References
    ----------
    .. [1] Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the
        Electronic Comptuer from the Point of View of the User."  Journal of
        the American Statistical Association.  62.319, 819-41.
        (https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat)
    """
    name = 'Longley'
    fname = name + '.csv'
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0)
    data = data.set_index('YEAR')

    # change period index to simple numeric index
    # TODO add support for period/datetime indexing
    # data.index = pd.PeriodIndex(data.index, freq='Y')
    data = data.reset_index(drop=True)

    # Get target series
    yname = 'TOTEMP'
    y = data.pop(yname)
    y = pd.Series([y], name=yname)

    # Get exogeneous series
    X = pd.DataFrame(
        [pd.Series([data.iloc[:, i]]) for i in range(data.shape[1])]).T
    X.columns = data.columns

    if return_X_y:
        y = y.iloc[0]
        return X, y
    else:
        X[yname] = y
        return X


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

    name = 'Lynx'
    fname = name + '.csv'
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0, squeeze=True)

    # change period index to simple numeric index
    # TODO add support for period/datetime indexing
    # data.index = pd.PeriodIndex(data.index, freq='Y')
    data = data.reset_index(drop=True)
    data.index = pd.Int64Index(data.index)
    data.name = name
    return data


def load_airline():
    """
    Load the airline univariate time series dataset for forecasting.

    Returns
    -------
    y : pandas Series
        Lynx sales dataset

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

    name = 'Airline'
    fname = name + '.csv'
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0, squeeze=True)

    # change period index to simple numeric index
    # TODO add support for period/datetime indexing
    # data.index = pd.PeriodIndex(data.index, freq='Y')
    data = data.reset_index(drop=True)
    data.index = pd.Int64Index(data.index)
    data.name = name
    return data
