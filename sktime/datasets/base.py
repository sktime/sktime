"""Utilities for loading datasets
"""
import os
import pandas as pd
from ..utils.load_data import load_from_tsfile_to_dataframe

__all__ = ["load_gunpoint",
           "load_arrow_head",
           "load_italy_power_demand",
           "load_shampoo_sales"]
__author__ = ['Markus Löning', 'Sajay Ganesh']

DIRNAME = 'data'
MODULE = os.path.dirname(__file__)


def _load_dataset(name, split, return_X_y):
    """Helper function to load datasets.
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
        return X, y
    else:
        X['class_val'] = pd.Series(y)
        return X


def load_gunpoint(split='ALL', return_X_y=False):
    """Loads the GunPoint time series classification problem and returns X and y

    Parameters
    ----------
    split: string (either "ALL" or "TRAIN" or "TEST", default = "ALL")
        Whether to load the train or test partition of the problem
        By default it loads both

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
    """Loads the ItalyPowerDemand time series classification problem and returns X and y

    Parameters
    ----------
    split: string (either "TRAIN" or "TEST", default = 'TRAIN')
        Whether to load the default train or test partition of the problem

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


def load_arrow_head(split='TRAIN', return_X_y=False):
    """Loads the ArrowHead time series classification problem and returns X and y.

    Parameters
    ----------
    split: string (either "TRAIN" or "TEST", default = 'TRAIN')
        Whether to load the default train or test partition of the problem

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


def load_shampoo_sales(return_dataframe=False):
    """Load the shampoo sales univariate time series forecasting dataset.

    Parameters
    ----------
    return_dataframe: bool
        - If True, returns pandas DataFrame.
        - If False, returns pandas Series.
        Default is False.

    Returns
    -------
    y : pandas Series/DataFrame
        Shampoo sales dataset

    Details
    -------
    This dataset describes the monthly number of sales of shampoo over a 3 year period. The units are a sales count.

    Dimensionality:     univariate
    Series length:      36
    Frequency:          Monthly
    Number of cases:    1

    References
    ----------
    Makridakis, Wheelwright and Hyndman (1998) Forecasting: methods and applications, John Wiley & Sons: New York.
    Chapter 3.
    """

    name = 'ShampooSales'
    fname = name + '.csv'
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0)
    data.index = pd.PeriodIndex(data.index, freq='M')
    if return_dataframe:
        # return nested pandas DataFrame with a single row and column
        return pd.DataFrame(pd.Series([pd.Series(data.squeeze())]), columns=[name])
    else:
        # return nested pandas Series with a single row
        return pd.Series([data.iloc[:, 0]], name=name)
