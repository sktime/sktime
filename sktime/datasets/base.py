'''Utilities for loading toy datasets for testing
'''
from os import path
import numpy as np
import pandas as pd


def read_single_series_data(f):
    '''reads data from the supplied file

    Parameters
    ----------
    file    : pre-opended file object
        the file to be read from

    Returns
    -------
    X       : A pandas DataFrame containing the time-series features
    y       : A pandas Series containing the targets
    '''
    data = f.readlines()
    rows = [row.strip().split('  ') for row in data]
    return pd.DataFrame(rows, dtype=np.float)


def load_gunpoint(split='TRAIN', return_X_y=False):
    '''Loads the gunpoint dataset as a pandas DataFrame

    This dataset involves one female actor and one male actor making a
    motion with their hand. The two classes are: Gun-Draw and Point:
    For Gun-Draw the actors have their hands by their sides. They draw
    a replicate gun from a hip-mounted holster, point it at a target
    for approximately one second, then return the gun to the holster,
    and their hands to their sides. For Point the actors have their gun
    by their sides. They point with their index fingers to a target for
    approximately one second, and then return their hands to their
    sides. For both classes, we tracked the centroid of the actor's
    right hands in both X- and Y-axes, which appear to be highly
    correlated. The data in the archive is just the X-axis.

    Parameters
    ----------
    split       : string (either "TRAIN" or "TEST"
        takes the default value "TRAIN". Used to specify the appropriate
        part of the dataset to be loaded.
    return_X_y  : bool
        If True, returns the features and target separately, each as pd.Series
        If False, returns both features and target in one DataFrame

    Returns
    -------
    X       : A pandas Series containing the time-series features
        Each entry in the series is a timeseries object
    y       : A pandas Series containing the targets
    '''
    module_path = path.dirname(__file__)
    dname = 'data'
    fname = 'GunPoint_'+split+'.txt'
    abspath = path.join(module_path, dname, fname)
    with open(abspath) as f:
        X = read_single_series_data(f)
    # remove the target before wrapping with series
    y = X.pop(0)
    y = y.astype(int)
    # create series of series
    X = pd.Series([np.array(row) for row in X.itertuples(index=False)])
    # set names for both series
    y.name = 'label'
    X.name = 'x_axis'
    # return as per user request
    if return_X_y:
        return X, y
    return pd.concat([X, y], axis=1)

"""Utilities for loading datasets for testing"""

from os import path
import numpy as np
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe


def load_gunpoint_dataframe(split='TRAIN', return_X_y=False):

    """Loads the GunPoint time series classification problem and returns X and y

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
    """
    module_path = path.dirname(__file__)
    dname = 'data'
    pname = 'GunPoint'
    fname = pname+'_'+split+'.ts'
    abspath = path.join(module_path, dname, pname, fname)

    X, y = load_from_tsfile_to_dataframe(abspath)
    if return_X_y:
        X['class_val'] = pd.Series(y)
        return X
    else:
        return X, y


def load_italy_power_demand_dataframe(split='TRAIN', return_X_y=False):
    """Loads the ItalyPowerDemand time series classification problem and returns X and y

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
    """

    module_path = path.dirname(__file__)
    dname = 'data'
    pname = 'ItalyPowerDemand'
    fname = pname+'_'+split+'.ts'
    abspath = path.join(module_path, dname, pname, fname)

    X, y = load_from_tsfile_to_dataframe(abspath)
    if return_X_y:
        X['class_val'] = pd.Series(y)
        return X
    else:
        return X, y


def load_arrow_head_dataframe(split='TRAIN', return_X_y=False):
    """Loads the ArrowHead time series classification problem and returns X and y

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
    """

    module_path = path.dirname(__file__)
    dname = 'data'
    pname = 'ArrowHead'
    fname = pname+'_'+split+'.ts'
    abspath = path.join(module_path, dname, pname, fname)

    X, y = load_from_tsfile_to_dataframe(abspath)
    if return_X_y:
        X['class_val'] = pd.Series(y)
        return X
    else:
        return X, y
