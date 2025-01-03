import numpy as np
import pandas as pd
import pytest

from sktime.datatypes._adapter.gluonts import (
    convert_pandas_collection_to_pandasDataset,
    convert_pandas_dataframe_to_pandasDataset,
    convert_pandas_multiindex_to_pandasDataset,
    convert_pandas_to_listDataset,
)
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("gluonts", severity="none"),
    reason="skip test if required soft dependency for GluonTS not available",
)
@pytest.mark.parametrize(
    "pandas_df",
    [
        (
            pd.DataFrame(
                {
                    "instances": np.zeros(50, dtype=int),
                    "timepoints": pd.date_range("2022-01-01", periods=50, freq="D"),
                    "target": np.random.randn(50),
                }
            )
        ),
        (
            pd.DataFrame(
                {
                    "instances": np.repeat(np.arange(3), 50),
                    "timepoints": np.tile(
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
    # Make the pandas DF multiindex
    pandas_df = pandas_df.set_index(["instances", "timepoints"])

    generated_list = convert_pandas_to_listDataset(pandas_df)
    idx = 0

    # Asserting equivalence for each time series in the ListDataset
    # and the series categories in the DataFrame
    for _, group_data in pandas_df.groupby(level=0):
        np.testing.assert_allclose(group_data.values, generated_list[idx]["target"])
        idx += 1

    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName

    gluon_dataset = []
    for entry in generated_list:
        gluon_dataset.append(
            {
                FieldName.START: entry["start"],
                FieldName.TARGET: entry["target"],
            }
        )

    # Attempt to create ListDataset object
    list_dataset = ListDataset(gluon_dataset, freq="D", one_dim_target=False)

    first_item = next(iter(list_dataset))
    assert "start" in first_item
    assert "target" in first_item
    np.testing.assert_allclose(first_item["target"], generated_list[0]["target"])


@pytest.mark.skipif(
    not _check_soft_dependencies("gluonts", severity="none"),
    reason="skip test if required soft dependency for GluonTS not available",
)
@pytest.mark.parametrize(
    "pandas_obj, conversion_function",
    [
        (
            pd.DataFrame(
                {
                    "instances": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                    "timepoints": pd.date_range("2022-01-01", periods=9, freq="D"),
                    "target": np.random.randn(9),
                },
            )
            .set_index(["instances", "timepoints"])
            .sort_index(),
            convert_pandas_multiindex_to_pandasDataset,
        ),
        (
            {
                "A": pd.DataFrame(
                    {
                        "timepoints": pd.date_range("2022-01-01", periods=50, freq="D"),
                        "target": np.random.randn(50),
                    }
                ),
                "B": pd.DataFrame(
                    {
                        "timepoints": pd.date_range("2022-01-01", periods=50, freq="D"),
                        "target": np.random.randn(50),
                    }
                ),
                "C": pd.DataFrame(
                    {
                        "timepoints": pd.date_range("2022-01-01", periods=50, freq="D"),
                        "target": np.random.randn(50),
                    }
                ),
            },
            convert_pandas_collection_to_pandasDataset,
        ),
        (
            pd.DataFrame(
                np.random.randn(50),
                index=pd.date_range("2022-01-01", periods=50, freq="D"),
            ),
            convert_pandas_dataframe_to_pandasDataset,
        ),
    ],
)
def test_pandas_df_to_PandasDataset(pandas_obj, conversion_function):
    # Attempting to convert the pandas object to a gluonTS PandasDataset
    conversion_function(pandas_obj)
