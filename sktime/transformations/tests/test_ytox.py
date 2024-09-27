import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.transformations.compose import YtoX
from sktime.transformations.series.exponent import ExponentTransformer


@pytest.fixture
def y_data():
    """Fixture to create a sample dataset."""
    y = load_airline()
    return y

def ensure_dataframe(output):
    """Ensure the output is a DataFrame."""
    if isinstance(output, pd.Series):
        return output.to_frame()
    return output

def test_ytox_with_transformer(y_data):
    """
    Test YtoX with a transformer applied, ensuring the result is the same
    as applying transformer directly to `y`.
    """
    X = pd.DataFrame(np.random.randn(len(y_data), 2), index=y_data.index)

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
        print("\nDiff between transformed via YtoX and directly transformed:\n", diff)

        raise
