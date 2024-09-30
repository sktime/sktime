import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.transformations.compose import YtoX


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


@pytest.mark.parametrize("params", YtoX.get_test_params())
def test_ytox_with_transformer(y_data, params):
    """
    Test YtoX with different transformers applied, ensuring the result is the same
    as applying transformer directly to `y`.
    """
    X = pd.DataFrame(np.random.randn(len(y_data), 2), index=y_data.index)

    # Extract transformer from params if available, otherwise set to None
    transformer = params.get("transformer", None)

    # Apply the transformer directly to y if it exists
    if transformer:
        transformed_y_direct = ensure_dataframe(transformer.fit_transform(y_data))
    else:
        transformed_y_direct = ensure_dataframe(y_data)  # No transformation

    # Create a YtoX transformer with the provided parameters
    ytox = YtoX(**params)

    # Fit YtoX first
    ytox.fit(X, y_data)

    # Apply the YtoX transformation
    transformed_y_via_ytox = ensure_dataframe(ytox.transform(X, y_data))

    # Check if the results are the same as applying the transformer directly to y
    pd.testing.assert_frame_equal(transformed_y_direct, transformed_y_via_ytox)

    diff = transformed_y_via_ytox.compare(transformed_y_direct)
    print("\nDiff between transformed via YtoX and directly transformed:\n", diff)
