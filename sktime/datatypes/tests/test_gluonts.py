import numpy as np
import pandas as pd
import pytest
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

from sktime.datatypes._table._convert import (
    convert_gluonts_listDataset_to_pandas,
    convert_pandas_to_gluonts_listDataset,
)


@pytest.mark.parametrize(
    "pandas_df",
    [
        (
            pd.DataFrame(
                {
                    "series_id": np.zeros(50, dtype=int),
                    "time": pd.date_range("2022-01-01", periods=50, freq="D"),
                    "target": np.random.randn(50),
                }
            )
        ),
        (
            pd.DataFrame(
                {
                    "series_id": np.repeat(np.arange(3), 50),
                    "time": np.tile(
                        pd.date_range("2022-01-01", periods=50, freq="D"), 3
                    ),
                    "0": np.random.randn(150),
                    "1": np.random.randn(150),
                    "2": np.random.randn(150),
                }
            )
        ),
    ],
)
def test_pandas_to_ListDataset(pandas_df):
    generated_list = convert_pandas_to_gluonts_listDataset(pandas_df)
    idx = 0

    # Asserting equivalence for each time series in the ListDataset
    # and the series categories in the DataFrame
    for _, group_data in pandas_df.groupby("series_id"):
        np.testing.assert_allclose(
            group_data.iloc[:, 2:].values, generated_list[idx]["target"]
        )

        idx += 1


@pytest.mark.parametrize(
    "list_dataset",
    [
        (
            ListDataset(
                [
                    {
                        FieldName.START: pd.Timestamp("2022-01-01"),
                        FieldName.TARGET: np.random.randn(50, 1),
                    }
                ],
                freq="D",
                one_dim_target=False,
            )
        ),
        (
            ListDataset(
                [
                    {
                        FieldName.START: pd.Timestamp("2022-01-01"),
                        FieldName.TARGET: np.random.randn(50, 5),
                    }
                ],
                freq="D",
                one_dim_target=False,
            )
        ),
    ],
)
def test_listDataset_to_pandas(list_dataset):
    pandas_df = convert_gluonts_listDataset_to_pandas(list_dataset)
    idx = 0

    # Asserting equivalence for each time series in the ListDataset
    # and the series categories in the DataFrame
    for _, group_data in pandas_df.groupby("series_id"):
        np.testing.assert_allclose(
            group_data.iloc[:, 2:].values, list_dataset[idx]["target"]
        )

        idx += 1
