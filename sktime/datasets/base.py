"""
Utilities for loading datasets
"""

import os
import pandas as pd

from ..utils.load_data import load_from_tsfile_to_dataframe

__all__ = ["load_gunpoint",
           "load_arrow_head",
           "load_italy_power_demand",
           "load_basic_motions",
           "load_shampoo_sales",
           "load_longley"]

__author__ = ['Markus Löning', 'Sajay Ganesh']

DIRNAME = 'data'
MODULE = os.path.dirname(__file__)


# time series classification data sets

def _load_dataset(name, split, return_X_y):
    """
    Helper function to load datasets.
    """

    if split in ["TRAIN", "TEST"]:
        fname = name + '_' + split + '.ts'
        abspath = os.path.join(MODULE, DIRNAME, name, fname)
        X, y = load_from_tsfile_to_dataframe(abspath)
    elif split == "ALL":
        X = pd.DataFrame()
        y = pd.Series()
        for split in ["TRAIN", "TEST"]:
            fname = name + '_' + split + '.ts'
            abspath = os.path.join(MODULE, DIRNAME, name, fname)
            result = load_from_tsfile_to_dataframe(abspath)
            X = pd.concat([X, pd.DataFrame(result[0])])
            y = pd.concat([y, pd.Series(result[1])])
    else:
        raise ValueError("Invalid split value")

    # Return appropriately
    if return_X_y:
        return (X, y)
    else:
        X['class_val'] = pd.Series(y)
        return X


def load_gunpoint(split='TRAIN', return_X_y=False):
    """
    Loads the GunPoint time series classification problem and returns X and y

    Parameters
    ----------
    split: str{"ALL", "TRAIN", "TEST"}, optional (default="TRAIN")
        Whether to load the train or test partition of the problem. By default it loads the train split.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single dataframe with columns for
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

    This dataset involves one female actor and one male actor making a motion with their
    hand. The two classes are: Gun-Draw and Point: For Gun-Draw the actors have their
    hands by their sides. They draw a replicate gun from a hip-mounted holster, point it
    at a target for approximately one second, then return the gun to the holster, and
    their hands to their sides. For Point the actors have their gun by their sides.
    They point with their index fingers to a target for approximately one second, and
    then return their hands to their sides. For both classes, we tracked the centroid
    of the actor's right hands in both X- and Y-axes, which appear to be highly
    correlated. The data in the archive is just the X-axis.

    Dataset details: http://timeseriesclassification.com/description.php?Dataset=GunPoint
    """
    name = 'GunPoint'
    return _load_dataset(name, split, return_X_y)


def load_italy_power_demand(split='TRAIN', return_X_y=False):
    """
    Loads the ItalyPowerDemand time series classification problem and returns X and y

    Parameters
    ----------
    split: str{"ALL", "TRAIN", "TEST"}, optional (default="TRAIN")
        Whether to load the train or test partition of the problem. By default it loads the train split.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single dataframe with columns for
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

    The data was derived from twelve monthly electrical power demand time series from Italy and
    first used in the paper "Intelligent Icons: Integrating Lite-Weight Data Mining and
    Visualization into GUI Operating Systems". The classification task is to distinguish days
    from Oct to March (inclusive) from April to September.

    Dataset details: http://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
    """

    name = 'ItalyPowerDemand'
    return _load_dataset(name, split, return_X_y)


def load_japanese_vowels(split='TRAIN', return_X_y=False):
    """
        Loads the JapaneseVowels time series classification problem and returns X and y.

        Parameters
        ----------
        split: str{"ALL", "TRAIN", "TEST"}, optional (default="TRAIN")
            Whether to load the train or test partition of the problem. By default it loads the train split.
        return_X_y: bool, optional (default=False)
            If True, returns (features, target) separately instead of a single dataframe with columns for
            features and the target.

        Returns
        -------
        X: pandas DataFrame with m rows and c columns
            The time series data for the problem with m cases and c dimensions
        y: numpy array
            The class labels for each case in X

        Details
        -------
        Dimensionality:     multivariate
        Series length:      29
        Train cases:        270
        Test cases:         370
        Number of classes:  9

        A UCI Archive dataset. 9 Japanese-male speakers were recorded saying the vowels 'a' and 'e'. A '12-degree
        linear prediction analysis' is applied to the raw recordings to obtain time-series with 12 dimensions, a
        originally a length between 7 and 29. In this dataset, instances have been padded to the longest length,
        29. The classification task is to predict the speaker. Therefore, each instance is a transformed utterance,
        12*29 values with a single class label attached, [1...9]. The given training set is comprised of 30
        utterances for each speaker, however the test set has a varied distribution based on external factors of
        timing and experimenal availability, between 24 and 88 instances per speaker. Reference: M. Kudo, J. Toyama
        and M. Shimbo. (1999). "Multidimensional Curve Classification Using Passing-Through Regions". Pattern
        Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.

        Dataset details: http://timeseriesclassification.com/description.php?Dataset=JapaneseVowels
    """

    name = 'JapaneseVowels'
    return _load_dataset(name, split, return_X_y)

def load_arrow_head(split='TRAIN', return_X_y=False):
    """
    Loads the ArrowHead time series classification problem and returns X and y.

    Parameters
    ----------
    split: str{"ALL", "TRAIN", "TEST"}, optional (default="TRAIN")
        Whether to load the train or test partition of the problem. By default it loads the train split.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single dataframe with columns for
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

    The arrowhead data consists of outlines of the images of arrowheads. The shapes of the
    projectile points are converted into a time series using the angle-based method. The
    classification of projectile points is an important topic in anthropology. The classes
    are based on shape distinctions such as the presence and location of a notch in the
    arrow. The problem in the repository is a length normalised version of that used in
    Ye09shapelets. The three classes are called "Avonlea", "Clovis" and "Mix"."

    Dataset details: http://timeseriesclassification.com/description.php?Dataset=ArrowHead
    """

    name = 'ArrowHead'
    return _load_dataset(name, split, return_X_y)


def load_basic_motions(split='TRAIN', return_X_y=False):
    """
    Loads the ArrowHead time series classification problem and returns X and y.

    Parameters
    ----------
    split: str{"ALL", "TRAIN", "TEST"}, optional (default="TRAIN")
        Whether to load the train or test partition of the problem. By default it loads the train split.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single dataframe with columns for
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

    The arrowhead data consists of outlines of the images of arrowheads. The shapes of the
    projectile points are converted into a time series using the angle-based method. The
    classification of projectile points is an important topic in anthropology. The classes
    are based on shape distinctions such as the presence and location of a notch in the
    arrow. The problem in the repository is a length normalised version of that used in
    Ye09shapelets. The three classes are called "Avonlea", "Clovis" and "Mix"."

    Dataset details: http://timeseriesclassification.com/description.php?Dataset=ArrowHead
    """

    name = 'BasicMotions'
    return _load_dataset(name, split, return_X_y)


# forecasting data sets

def load_shampoo_sales(return_y_as_dataframe=False):
    """
    Load the shampoo sales univariate time series forecasting dataset.

    Parameters
    ----------
    return_y_as_dataframe: bool, optional (default=False)
        Whether to return target series as series or dataframe, useful for high-level interface.
        - If True, returns target series as pandas.DataFrame.s
        - If False, returns target series as pandas.Series.

    Returns
    -------
    y : pandas Series/DataFrame
        Shampoo sales dataset

    Details
    -------
    This dataset describes the monthly number of sales of shampoo over a 3 year period.
    The units are a sales count.

    Dimensionality:     univariate
    Series length:      36
    Frequency:          Monthly
    Number of cases:    1


    References
    ----------
    ..[1] Makridakis, Wheelwright and Hyndman (1998) Forecasting: methods and applications,
        John Wiley & Sons: New York. Chapter 3.
    """

    name = 'ShampooSales'
    fname = name + '.csv'
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0)
    data.index = pd.PeriodIndex(data.index, freq='M')
    if return_y_as_dataframe:
        # return nested pandas DataFrame with a single row and column
        return pd.DataFrame(pd.Series([pd.Series(data.squeeze())]), columns=[name])
    else:
        # return nested pandas Series with a single row
        return pd.Series([data.iloc[:, 0]], name=name)


def load_longley(return_X_y=False, return_y_as_dataframe=False):
    """
    Load the Longley dataset for forecasting with exogenous variables.


    Parameters
    ----------
    return_y_as_dataframe: bool, optional (default=False)
        Whether to return target series as series or dataframe, useful for high-level interface.
        - If True, returns target series as pandas.DataFrame.s
        - If False, returns target series as pandas.Series.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas.DataFrame
        The exogenous time series data for the problem.
    y: pandas.Series
        The target series to be predicted.

    Details
    -------
    This dataset contains various US macroeconomic variables from 1947 to 1962 that are known to be highly
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
    ..[1] Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the
        Electronic Comptuer from the Point of View of the User."  Journal of
        the American Statistical Association.  62.319, 819-41.
        (https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat)
    """

    if return_y_as_dataframe and not return_X_y:
        raise ValueError("`return_y_as_dataframe` can only be set to True if `return_X_y` is True, "
                         "otherwise y is given as a column in the returned dataframe and "
                         "cannot be returned as a separate dataframe.")

    name = 'Longley'
    fname = name + '.csv'
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0)
    data = data.set_index('YEAR')
    data.index = pd.PeriodIndex(data.index, freq='Y')

    # Get target series
    yname = 'TOTEMP'
    y = data.pop(yname)
    y = pd.Series([y], name=yname)

    # Get feature series
    X = pd.DataFrame([pd.Series([data.iloc[:, i]]) for i in range(data.shape[1])]).T
    X.columns = data.columns

    if return_X_y:
        if return_y_as_dataframe:
            y = pd.DataFrame(pd.Series([pd.Series(y.squeeze())]), columns=[yname])
            return X, y
        else:
            return X, y
    else:
        X[yname] = y
        return X
