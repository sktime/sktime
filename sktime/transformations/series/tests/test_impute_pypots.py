# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of PyPOTSImputer functionality."""

import numpy as np
import pytest

from sktime.transformations.series.impute import PyPOTSImputer
from sktime.utils._testing.forecasting import make_forecasting_problem


@pytest.mark.parametrize("model", ["Lerp", "LOCF"])
def test_pypots_imputer_basic(model):
    """Test PyPOTSImputer with basic statistical models."""
    y = make_forecasting_problem(make_X=False)
    # y is Series if make_X=False, or DataFrame?
    # Actually make_forecasting_problem returns (y, X) if make_X=True
    # and returns y if make_X=False.
    y.iloc[3:6] = np.nan
    
    transformer = PyPOTSImputer(model=model)
    y_hat = transformer.fit_transform(y)
    
    assert not y_hat.isnull().to_numpy().any()
    assert y_hat.shape == y.shape


def test_pypots_imputer_multivariate():
    """Test PyPOTSImputer with multivariate data."""
    y, X = make_forecasting_problem(make_X=True)
    X.iloc[2, 0] = np.nan
    X.iloc[4, 1] = np.nan
    
    transformer = PyPOTSImputer(model="Lerp")
    X_hat = transformer.fit_transform(X)
    
    assert not X_hat.isnull().to_numpy().any()
    assert X_hat.shape == X.shape
