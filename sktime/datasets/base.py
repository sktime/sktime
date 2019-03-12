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
    # create series of series
    X = pd.Series([np.array(row) for row in X.itertuples(index=False)])
    # set names for both series
    y.name = 'target'
    X.name = 'x_axis'
    # return as per user request
    if return_X_y:
        return X, y
    return pd.concat([X, y], axis=1)
