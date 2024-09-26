import pytest
import pandas as pd
import numpy as np
from sktime.datasets import load_airline
from sktime.transformations.compose import YtoX
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.boxcox import BoxCoxTransformer


@pytest.fixture
def y_data():
    """Fixture to create a sample dataset."""
    y = load_airline()
    return y


def get_test_params():
    """Provide test parameters for the YtoX class."""
    # Return instances with different parameters
    return [
        {"ytox": YtoX(subset_index=False), "description": "YtoX without subset_index"},
        {"ytox": YtoX(subset_index=True), "description": "YtoX with subset_index"},
        {"ytox": YtoX(subset_index=False, transformer=ExponentTransformer(power=2)), "description": "YtoX with ExponentTransformer"},
        {"ytox": YtoX(subset_index=False, transformer=BoxCoxTransformer()), "description": "YtoX with BoxCoxTransformer"},
    ]


def ensure_dataframe(output):
    """Ensure the output is a DataFrame."""
    if isinstance(output, pd.Series):
        return output.to_frame()
    return output


@pytest.mark.parametrize("params", get_test_params())
def test_ytox(params, y_data):
    """Test YtoX with different parameters."""
    ytox = params["ytox"]
    description = params["description"]

    X = pd.DataFrame(np.random.randn(len(y_data), 2), index=y_data.index)  # Example exogenous data

    # Call fit first
    ytox.fit(X, y_data)

    transformed_y = ensure_dataframe(ytox.transform(X, y_data))

    # Ensure both are DataFrames
    if ytox.transformer is not None:
        expected_y = ensure_dataframe(ytox.transformer.fit_transform(y_data))
    else:
        expected_y = ensure_dataframe(y_data)

    try:
        # Check if transformed data is the same as the original `y`
        pd.testing.assert_frame_equal(transformed_y, expected_y)
    except AssertionError as e:
        print(f"\nError in {description}:")
        print("\nTransformed y:\n", transformed_y.head())
        print("\nExpected y:\n", expected_y.head())
        print("\nError:\n", e)

        # Perform a detailed comparison to find exact differences
        diff = transformed_y.compare(expected_y)
        print("\nDifferences between transformed and expected:\n", diff)

        raise


def test_ytox_with_transformer(y_data):
    """Test YtoX with a transformer applied, ensuring the result is the same as applying transformer directly to `y`."""
    X = pd.DataFrame(np.random.randn(len(y_data), 2), index=y_data.index)  # Example exogenous data

    # Use an example transformer, e.g., an exponent transformer
    transformer = ExponentTransformer(power=2)

    # Apply the transformer directly to y
    transformed_y_direct = ensure_dataframe(transformer.fit_transform(y_data))

    # Create a YtoX transformer with the provided transformer
    ytox = YtoX(transformer=transformer)

    # Fit YtoX first
    ytox.fit(X, y_data)

    # Apply the YtoX transformation
    transformed_y_via_ytox = ensure_dataframe(ytox.transform(X, y_data))

    try:
        # Check if the results are the same as applying the transformer directly to y
        pd.testing.assert_frame_equal(transformed_y_direct, transformed_y_via_ytox)
    except AssertionError as e:
        print("\nTransformed y via YtoX:\n", transformed_y_via_ytox.head())
        print("\nDirectly Transformed y:\n", transformed_y_direct.head())
        print("\nError:\n", e)

        # Perform a detailed comparison to find exact differences
        diff = transformed_y_via_ytox.compare(transformed_y_direct)
        print("\nDifferences between transformed via YtoX and directly transformed:\n", diff)

        raise


@pytest.mark.parametrize("params", get_test_params())
def test_ytox_subset_index(params, y_data):
    """Test the subset_index option of YtoX."""
    ytox = params["ytox"]
    description = params["description"]

    # Ensure the YtoX transformer behaves as expected with subset_index=True
    X = pd.DataFrame(np.random.randn(len(y_data), 2), index=y_data.index)  # Example exogenous data

    # Subset the X to a smaller index
    X_subset = X.iloc[:-5]

    # Call fit first
    ytox.fit(X_subset, y_data.iloc[:-5])

    transformed_y = ensure_dataframe(ytox.transform(X_subset, y_data.iloc[:-5]))

    # Ensure both are DataFrames
    if ytox.transformer is not None:
        expected_y = ensure_dataframe(ytox.transformer.fit_transform(y_data.iloc[:-5]))
    else:
        expected_y = ensure_dataframe(y_data.iloc[:-5])

    try:
        # Check if transformed data is the same as the original `y`
        transformed_y.reset_index(drop=True, inplace=True)
        expected_y.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(transformed_y, expected_y)
    except AssertionError as e:
        print(f"\nError in {description}:")
        print("\nTransformed y:\n", transformed_y.head())
        print("\nExpected y:\n", expected_y.head())
        print("\nError:\n", e)

        # Perform a detailed comparison to find exact differences
        diff = transformed_y.compare(expected_y)
        print("\nDifferences between transformed and expected:\n", diff)

        raise