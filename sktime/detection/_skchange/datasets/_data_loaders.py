"""Functions for loading datasets that ships with `skchange`."""

import os

import pandas as pd


def load_hvac_system_data():
    """Load the heating, ventilation and air conditioning (HVAC) system dataset.

    The dataset contains time series of vibration magnitude measurements from two
    different HVAC systems. 30 days of sensor measurements are available for each unit,
    with a sampling rate of 10 minutes.

    The aim of analysing the data is to detect when each hvac system normally
    turns on and off, such that anomalies from their regular schedule can be detected.
    True labels are not available, but it is fairly easy to identify the normal
    schedule from the data and observe when the units deviate from it.

    The data has been provided by the company Soundsensing:
    https://www.soundsensing.no/.

    Returns
    -------
    pd.DataFrame
        The HVAC system dataset. It has a multi-index with two levels:

        1. "unit_id": A string identifier for each hvac unit.
        2. "time": A datetime index with the time of each measurement.

        There's one column:

        1. "vibration": A float column with the vibration magnitude measurements.
    """
    this_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(this_file_dir, "data", "hvac_system", "data.csv")
    df = pd.read_csv(file_path).iloc[:, 1:]
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["unit_id", "time"])
    return df
