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
    X = pd.DataFrame(rows, dtype=np.float)
    y = X.pop(0)
    return X, y


def load_gunpoint(split='TRAIN'):
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
    split   : string (either "TRAIN" or "TEST"
        takes the default value "TRAIN". Used to specify the appropriate
        part of the dataset to be loaded.

    Returns
    -------
    X       : A pandas Series containing the time-series features
        Each entry in the series is a timeseries object
    y       : A pandas Series containing the targets
    '''
    module_path = path.dirname(__file__)
    dname = 'datasets'
    fname = 'GunPoint_'+split+'.txt'
    abspath = path.join(module_path, dname, fname)
    with open(abspath) as f:
        X, y = read_single_series_data(f)
    # create series of series
    X = pd.Series([row for _, row in X.iterrows()])
    return X, y
