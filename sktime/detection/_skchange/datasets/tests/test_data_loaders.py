import pandas as pd

from sktime.detection._skchange.datasets import load_hvac_system_data


def test_load_hvac_system_data():
    df = load_hvac_system_data()

    # Check if the returned object is a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Check if the DataFrame has a MultiIndex with levels "unit_id" and "time"
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["unit_id", "time"]

    # Check if the DataFrame has one column named "vibration"
    assert list(df.columns) == ["vibration"]

    # Check if the "vibration" column is of float type
    assert pd.api.types.is_float_dtype(df["vibration"])
