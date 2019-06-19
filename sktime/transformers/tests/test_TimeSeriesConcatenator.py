from sktime.transformers.series_to_series import TimeSeriesConcatenator
from sktime.datasets import load_gunpoint
import pandas as pd
import pytest


@pytest.mark.parametrize("n_dims", [1, 3, 5])
def test_TimeSeriesConcatenator(n_dims):

    univariate, y = load_gunpoint(return_X_y=True)
    multivariate = pd.concat([univariate] * n_dims, axis=1)

    trans = TimeSeriesConcatenator()

    Xt = trans.fit_transform(multivariate)

    # check if transformed dataframe is univariate
    assert Xt.shape[1] == 1

    # check if number of time series observations are correct
    assert Xt.iloc[0, 0].shape[0] == univariate.iloc[0, 0].shape[0] * n_dims


